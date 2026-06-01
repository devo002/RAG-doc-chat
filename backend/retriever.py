import json
from typing import AsyncIterator

from qdrant_client.models import Filter, FieldCondition, MatchAny

import state
from config import TOP_K_RETRIEVE, TOP_K_RERANK, COLLECTION_NAME

CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based strictly on the provided document sections. "
    "Each source label includes the filename and page number, e.g. [Source 1 | report.pdf | Page 5]. "
    "When citing, always include the page number in your citation, e.g. [Source 1, Page 5]. "
    "If the answer is not found in the document sections provided, say so clearly. "
    "Be precise and concise. "
    "Do not use markdown formatting — no ##, no **, no bullet dashes. Write in plain prose."
)

EVAL_SYSTEM_PROMPT = (
    "You are a precise assistant. Answer in 1-2 sentences maximum, based strictly on the provided document sections. "
    "If the answer is not in the context, say 'Not found in document.'"
)


def embed(texts: list[str]) -> list[list[float]]:
    return [vec.tolist() for vec in state.embedder.embed(texts)]


def _search(q_vec: list[float], doc_ids: list[str] | None, limit: int):
    query_filter = None
    if doc_ids:
        query_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))]
        )
    response = state.qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )
    return response.points


def _build_context(results):
    top = results[:TOP_K_RERANK]
    context_parts, sources = [], []
    for rank_i, hit in enumerate(top):
        p = hit.payload
        page_label = "Cover/Abstract" if p["page"] == 0 else f"Page {p['page']}"
        doc_text = p["text"]
        context_parts.append(f"[Source {rank_i + 1} | {p['filename']} | {page_label}]\n{doc_text}")
        sources.append({
            "rank": rank_i + 1,
            "filename": p["filename"],
            "page": p["page"],
            "doc_id": p["doc_id"],
            "similarity": round(float(hit.score), 4),
            "snippet": doc_text[:220] + ("…" if len(doc_text) > 220 else ""),
        })
    return "\n\n---\n\n".join(context_parts), sources


async def rag_stream(query: str, doc_ids: list[str] | None) -> AsyncIterator[str]:
    q_vec = embed([query])[0]
    results = _search(q_vec, doc_ids, TOP_K_RETRIEVE)

    if not results:
        yield "data: " + json.dumps({"type": "error", "content": "No relevant content found."}) + "\n\n"
        return

    context, sources = _build_context(results)
    user_message = (
        f"Context from uploaded documents:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}\n\n"
        "Answer based on the context above, citing [Source N, Page X] for each claim:"
    )

    yield "data: " + json.dumps({"type": "sources", "sources": sources}) + "\n\n"

    with state.anthropic_client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=CHAT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text_chunk in stream.text_stream:
            yield "data: " + json.dumps({"type": "token", "content": text_chunk}) + "\n\n"

    yield "data: " + json.dumps({"type": "done"}) + "\n\n"


def rag_answer(query: str, doc_ids: list[str] | None = None) -> dict:
    q_vec = embed([query])[0]
    results = _search(q_vec, doc_ids, TOP_K_RETRIEVE)

    if not results:
        return {"answer": "No relevant content found.", "context": "", "retrieved_count": 0}

    context, _ = _build_context(results)
    user_message = (
        f"Context from uploaded documents:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}\n\nAnswer in 1-2 sentences:"
    )

    response = state.anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        system=EVAL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return {"answer": response.content[0].text, "context": context, "retrieved_count": len(results)}
