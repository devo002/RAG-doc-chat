import re
import json
import numpy as np
from pathlib import Path

import state
from retriever import rag_answer, _search, _hybrid_fuse, _rerank, embed
from config import TOP_K_RETRIEVE, TOP_K_RERANK

_eval_data = json.loads((Path(__file__).parent / "eval_data.json").read_text())
TEST_QUESTIONS = [{"id": i + 1, **item} for i, item in enumerate(_eval_data)]


def ragas_score(question: str, context: str, answer: str, ground_truth: str) -> dict:
    prompt = (
        f"You are an evaluation judge for a RAG system.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"Generated Answer: {answer}\n\n"
        f"Ground Truth Answer: {ground_truth}\n\n"
        f"Score these three metrics from 0.0 to 1.0:\n"
        f"- faithfulness: fraction of answer statements directly supported by the retrieved context "
        f"(1.0 = fully grounded, 0.0 = not supported at all)\n"
        f"- answer_relevancy: how well the answer addresses the question "
        f"(1.0 = perfectly on-topic, 0.0 = off-topic)\n"
        f"- correctness: how factually correct the generated answer is compared to the ground truth "
        f"(1.0 = same key facts, 0.0 = contradicts or missing key facts)\n\n"
        f"Return ONLY valid JSON, no explanation:\n"
        f'{{\"faithfulness\": <float>, \"answer_relevancy\": <float>, \"correctness\": <float>}}'
    )
    response = state.anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=80,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        parsed = json.loads(match.group())
        return {
            "faithfulness": round(float(parsed.get("faithfulness", 0)), 3),
            "answer_relevancy": round(float(parsed.get("answer_relevancy", 0)), 3),
            "correctness": round(float(parsed.get("correctness", 0)), 3),
        }
    return {"faithfulness": 0.0, "answer_relevancy": 0.0, "correctness": 0.0}


def _modality_counts(question: str, doc_ids: list[str] | None) -> dict:
    """Count how many text vs image chunks appear in the top-K retrieved for this question."""
    try:
        q_vec = embed([question])[0]
        hits = _search(q_vec, doc_ids, TOP_K_RETRIEVE)
        reranked = _rerank(question, _hybrid_fuse(question, hits))
        top = reranked[:TOP_K_RERANK]
        text_count = sum(1 for h in top if h.payload.get("chunk_type", "text") == "text")
        image_count = sum(1 for h in top if h.payload.get("chunk_type") == "image")
        return {"text_chunks": text_count, "image_chunks": image_count}
    except Exception:
        return {"text_chunks": 0, "image_chunks": 0}


def run_evaluation_stream(doc_ids: list[str] | None = None):
    results = []
    total = len(TEST_QUESTIONS)
    for tq in TEST_QUESTIONS:
        result = rag_answer(tq["question"], doc_ids)
        scores = ragas_score(tq["question"], result["context"], result["answer"], tq["ground_truth"])
        ragas = round((scores["faithfulness"] + scores["answer_relevancy"] + scores["correctness"]) / 3, 3)
        modality = _modality_counts(tq["question"], doc_ids)
        q_result = {
            "id": tq["id"],
            "question": tq["question"],
            "answer": result["answer"],
            "ground_truth": tq["ground_truth"],
            "faithfulness": scores["faithfulness"],
            "answer_relevancy": scores["answer_relevancy"],
            "correctness": scores["correctness"],
            "ragas_score": ragas,
            "retrieved_count": result["retrieved_count"],
            "text_chunks": modality["text_chunks"],
            "image_chunks": modality["image_chunks"],
            "total": total,
        }
        results.append(q_result)
        yield f"data: {json.dumps({'type': 'question', 'result': q_result})}\n\n"

    overall = {
        "avg_faithfulness": round(float(np.mean([r["faithfulness"] for r in results])), 3),
        "avg_answer_relevancy": round(float(np.mean([r["answer_relevancy"] for r in results])), 3),
        "avg_correctness": round(float(np.mean([r["correctness"] for r in results])), 3),
        "avg_ragas_score": round(float(np.mean([r["ragas_score"] for r in results])), 3),
        "avg_image_chunks": round(float(np.mean([r["image_chunks"] for r in results])), 2),
    }
    yield f"data: {json.dumps({'type': 'done', 'overall': overall})}\n\n"
