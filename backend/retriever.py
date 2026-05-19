import json
from typing import AsyncIterator

import state
from config import TOP_K_RETRIEVE, TOP_K_RERANK

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


def _query_collection(q_vec, doc_ids: list[str] | None, include: list[str]) -> dict:
    where_filter = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    n = min(TOP_K_RETRIEVE, state.collection.count() or 1)
    return state.collection.query(
        query_embeddings=[q_vec],
        n_results=n,
        where=where_filter,
        include=include,
    )


def _build_context(docs: list, metas: list, dists: list | None = None):
    top = list(zip(docs, metas, dists or [None] * len(docs)))[:TOP_K_RERANK]
    context_parts, sources = [], []
    for rank_i, (doc_text, meta, dist) in enumerate(top):
        page_label = "Cover/Abstract" if meta["page"] == 0 else f"Page {meta['page']}"
        context_parts.append(f"[Source {rank_i + 1} | {meta['filename']} | {page_label}]\n{doc_text}")
        if dist is not None:
            sources.append({
                "rank": rank_i + 1,
                "filename": meta["filename"],
                "page": meta["page"],
                "doc_id": meta["doc_id"],
                "similarity": round(float(1 - dist), 4),
                "cosine_distance": round(float(dist), 4),
                "snippet": doc_text[:220] + ("…" if len(doc_text) > 220 else ""),
            })
    return "\n\n---\n\n".join(context_parts), sources


async def rag_stream(query: str, doc_ids: list[str] | None) -> AsyncIterator[str]:
    q_vec = embed([query])[0]
    results = _query_collection(q_vec, doc_ids, ["documents", "metadatas", "distances"])

    docs_raw  = results["documents"][0]
    metas_raw = results["metadatas"][0]
    dists_raw = results["distances"][0]

    if not docs_raw:
        yield "data: " + json.dumps({"type": "error", "content": "No relevant content found."}) + "\n\n"
        return

    context, sources = _build_context(docs_raw, metas_raw, dists_raw)
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
    results = _query_collection(q_vec, doc_ids, ["documents", "metadatas", "distances"])

    docs_raw  = results["documents"][0]
    metas_raw = results["metadatas"][0]

    if not docs_raw:
        return {"answer": "No relevant content found.", "context": "", "retrieved_count": 0}

    context, _ = _build_context(docs_raw, metas_raw)
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
    return {"answer": response.content[0].text, "context": context, "retrieved_count": len(docs_raw)}
