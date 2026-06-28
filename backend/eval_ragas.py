"""
Offline RAGAS-style evaluation script.

Works against local OR production backend — set BACKEND_URL env var:

  Local:
      BACKEND_URL=http://localhost:8000 python eval_ragas.py

  Production (Railway):
      BACKEND_URL=https://rag-doc-chat-production.up.railway.app python eval_ragas.py

  Or pass as a flag:
      python eval_ragas.py --base-url https://rag-doc-chat-production.up.railway.app

  Target a specific document:
      python eval_ragas.py --doc-id <id>

Requires ANTHROPIC_API_KEY to be set.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic
import requests

DEFAULT_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
EVAL_DATA = Path(__file__).parent / "eval_data.json"
HAIKU = "claude-haiku-4-5-20251001"


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def get_documents(base_url: str) -> list[dict]:
    r = requests.get(f"{base_url}/documents", timeout=15)
    r.raise_for_status()
    return r.json()


def call_rag(question: str, doc_id: str, base_url: str) -> tuple[str, str]:
    """Stream /chat and return (answer, context_snippets)."""
    r = requests.post(
        f"{base_url}/chat",
        json={"query": question, "doc_ids": [doc_id]},
        stream=True,
        timeout=90,
    )
    r.raise_for_status()

    answer = ""
    sources = []
    for raw in r.iter_lines():
        if not raw or not raw.startswith(b"data: "):
            continue
        event = json.loads(raw[6:])
        if event["type"] == "token":
            answer += event["content"]
        elif event["type"] == "sources":
            sources = event["sources"]
        elif event["type"] == "error":
            raise RuntimeError(event["content"])

    context = "\n\n---\n\n".join(s["snippet"] for s in sources)
    return answer.strip(), context


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_faithfulness(client: anthropic.Anthropic, answer: str, context: str) -> tuple[float, list]:
    """
    Decompose answer into claims, verify each against retrieved context.
    Score = supported_claims / total_claims.
    """
    prompt = (
        f"Retrieved context:\n{context}\n\n"
        f"Answer to evaluate:\n{answer}\n\n"
        "1. List every factual claim in the answer.\n"
        "2. For each claim state whether the retrieved context above directly supports it.\n"
        'Return ONLY valid JSON: {"claims": [{"claim": "...", "supported": true}]}'
    )
    resp = client.messages.create(
        model=HAIKU,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        data = json.loads(resp.content[0].text.strip())
        claims = data["claims"]
        score = sum(1 for c in claims if c["supported"]) / len(claims) if claims else 0.0
        return round(score, 3), claims
    except Exception:
        return 0.0, []


def score_answer_relevancy(client: anthropic.Anthropic, answer: str, question: str) -> float:
    """
    Generate reverse questions from the answer, then score how well
    they match the original question. Score = 0-1.
    """
    prompt = (
        f'Answer: "{answer}"\n\n'
        "Generate 3 questions that this answer could be directly responding to.\n"
        'Return ONLY valid JSON: {"questions": ["q1", "q2", "q3"]}'
    )
    resp = client.messages.create(
        model=HAIKU,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        generated = json.loads(resp.content[0].text.strip())["questions"]
    except Exception:
        return 0.0

    sim_prompt = (
        f'Original question: "{question}"\n\n'
        "Generated questions:\n" + "\n".join(f"- {q}" for q in generated) + "\n\n"
        "Score the average semantic similarity between the generated questions and the original "
        "(0.0 = completely unrelated, 1.0 = same meaning).\n"
        'Return ONLY valid JSON: {"score": 0.0}'
    )
    sim_resp = client.messages.create(
        model=HAIKU,
        max_tokens=50,
        messages=[{"role": "user", "content": sim_prompt}],
    )
    try:
        return round(float(json.loads(sim_resp.content[0].text.strip())["score"]), 3)
    except Exception:
        return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", default=None, help="Doc ID to evaluate against")
    parser.add_argument("--base-url", default=DEFAULT_URL, help="Backend base URL")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set ANTHROPIC_API_KEY before running.")

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Connecting to backend: {args.base_url}")
    try:
        docs = get_documents(args.base_url)
    except Exception as e:
        sys.exit(f"ERROR: Could not reach backend — {e}")

    if not docs:
        sys.exit("No documents indexed. Upload a PDF first.")

    if args.doc_id:
        doc_id = args.doc_id
        doc_name = next((d["filename"] for d in docs if d["doc_id"] == doc_id), doc_id)
    else:
        doc_id = docs[0]["doc_id"]
        doc_name = docs[0]["filename"]

    test_set = json.loads(EVAL_DATA.read_text())

    print(f"Document : {doc_name}")
    print(f"Questions: {len(test_set)}")
    print("=" * 72)

    results = []
    for i, item in enumerate(test_set, 1):
        question = item["question"]
        print(f"\n[{i}/{len(test_set)}] {question}")

        try:
            answer, context = call_rag(question, doc_id, args.base_url)
        except Exception as e:
            print(f"  Pipeline error: {e}")
            continue

        faith_score, claims = score_faithfulness(client, answer, context)
        rel_score = score_answer_relevancy(client, answer, question)
        ragas_score = round((faith_score + rel_score) / 2, 3)

        results.append({
            "question": question,
            "ground_truth": item["ground_truth"],
            "answer": answer,
            "faithfulness": faith_score,
            "answer_relevancy": rel_score,
            "ragas_score": ragas_score,
            "claims": claims,
        })

        print(f"  Faithfulness:     {faith_score:.3f}")
        print(f"  Answer Relevancy: {rel_score:.3f}")
        print(f"  RAGAS Score:      {ragas_score:.3f}")

    if not results:
        print("\nNo results — check that the backend is running and the document is indexed.")
        return

    avg_faith = round(sum(r["faithfulness"] for r in results) / len(results), 3)
    avg_rel   = round(sum(r["answer_relevancy"] for r in results) / len(results), 3)
    avg_ragas = round(sum(r["ragas_score"] for r in results) / len(results), 3)

    print("\n" + "=" * 72)
    print(f"SUMMARY  ({len(results)}/{len(test_set)} questions)")
    print(f"  Avg Faithfulness:     {avg_faith:.3f}")
    print(f"  Avg Answer Relevancy: {avg_rel:.3f}")
    print(f"  Avg RAGAS Score:      {avg_ragas:.3f}")
    print("=" * 72)

    out = Path(__file__).parent / "eval_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved → {out}")


if __name__ == "__main__":
    main()
