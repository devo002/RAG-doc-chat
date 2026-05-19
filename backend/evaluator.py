import re
import json
import numpy as np

import state
from retriever import rag_answer

TEST_QUESTIONS = [
    {"id": 1, "question": "What is this document about? Summarise in one sentence."},
    {"id": 2, "question": "Who is the author of this document?"},
    {"id": 3, "question": "What are the main results or conclusions of this research?"},
    {"id": 4, "question": "What methods or techniques are used in this study?"},
    {"id": 5, "question": "What data or evidence is presented to support the main claims?"},
]


def ragas_score(question: str, context: str, answer: str) -> dict:
    prompt = (
        f"You are an evaluation judge for a RAG system.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"Answer: {answer}\n\n"
        f"Score these two metrics from 0.0 to 1.0:\n"
        f"- faithfulness: fraction of answer statements directly supported by the context "
        f"(1.0 = fully grounded, 0.0 = not supported)\n"
        f"- answer_relevancy: how well the answer addresses the question "
        f"(1.0 = perfectly on-topic, 0.0 = off-topic)\n\n"
        f"Return ONLY valid JSON, no explanation:\n"
        f'{{\"faithfulness\": <float>, \"answer_relevancy\": <float>}}'
    )
    response = state.anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=60,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        parsed = json.loads(match.group())
        return {
            "faithfulness": round(float(parsed.get("faithfulness", 0)), 3),
            "answer_relevancy": round(float(parsed.get("answer_relevancy", 0)), 3),
        }
    return {"faithfulness": 0.0, "answer_relevancy": 0.0}


def run_evaluation(doc_ids: list[str] | None = None) -> dict:
    results = []
    for tq in TEST_QUESTIONS:
        result = rag_answer(tq["question"], doc_ids)
        scores = ragas_score(tq["question"], result["context"], result["answer"])
        ragas = round((scores["faithfulness"] + scores["answer_relevancy"]) / 2, 3)
        results.append({
            "id": tq["id"],
            "question": tq["question"],
            "answer": result["answer"],
            "faithfulness": scores["faithfulness"],
            "answer_relevancy": scores["answer_relevancy"],
            "ragas_score": ragas,
            "retrieved_count": result["retrieved_count"],
        })
    overall = {
        "avg_faithfulness": round(float(np.mean([r["faithfulness"] for r in results])), 3),
        "avg_answer_relevancy": round(float(np.mean([r["answer_relevancy"] for r in results])), 3),
        "avg_ragas_score": round(float(np.mean([r["ragas_score"] for r in results])), 3),
    }
    return {"questions": results, "overall": overall}
