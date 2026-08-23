import json
import math
from collections import Counter
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


def _bm25_scores(query: str, hits: list, k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Pure-Python BM25Okapi — no extra dependencies, negligible memory."""
    corpus = [h.payload["text"].lower().split() for h in hits]
    q_tokens = query.lower().split()
    N = len(corpus)
    avg_dl = sum(len(d) for d in corpus) / N if N else 1
    scores = []
    for doc in corpus:
        dl = len(doc)
        tf_map = Counter(doc)
        score = 0.0
        for term in q_tokens:
            tf = tf_map.get(term, 0)
            df = sum(1 for d in corpus if term in Counter(d))
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
        scores.append(score)
    return scores


def _hybrid_fuse(query: str, hits: list) -> list:
    """Reciprocal Rank Fusion of dense + BM25 rankings (k=60 standard constant)."""
    if not hits:
        return hits
    bm25 = _bm25_scores(query, hits)
    bm25_order = sorted(range(len(hits)), key=lambda i: bm25[i], reverse=True)
    bm25_rank = {orig: rank for rank, orig in enumerate(bm25_order)}
    K = 60
    rrf = [1 / (K + i) + 1 / (K + bm25_rank[i]) for i in range(len(hits))]
    return [hits[i] for i in sorted(range(len(hits)), key=lambda i: rrf[i], reverse=True)]


def _rerank(query: str, results: list) -> list:
    if len(results) <= TOP_K_RERANK:
        return results
    chunks = "\n\n".join(
        f"[{i}] {hit.payload['text'][:400]}" for i, hit in enumerate(results)
    )
    prompt = (
        f'Query: "{query}"\n\n'
        f"Rank these {len(results)} chunks by relevance to the query.\n"
        f"Return ONLY a JSON array of indices, most relevant first. Example: [3,0,7,2,1]\n\n"
        f"{chunks}\n\nJSON array:"
    )
    try:
        response = state.anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        indices = json.loads(response.content[0].text.strip())
        seen: set[int] = set()
        valid = [
            i for i in indices
            if isinstance(i, int) and 0 <= i < len(results) and not (i in seen or seen.add(i))  # type: ignore[func-returns-value]
        ]
        missing = [i for i in range(len(results)) if i not in seen]
        return [results[i] for i in valid + missing]
    except Exception:
        return results


def _build_context(results):
    top = results[:TOP_K_RERANK]
    context_parts, content_blocks, sources = [], [], []

    for rank_i, hit in enumerate(top):
        p = hit.payload
        page_label = "Cover/Abstract" if p["page"] == 0 else f"Page {p['page']}"
        chunk_type = p.get("chunk_type", "text")
        label = f"[Source {rank_i + 1} | {p['filename']} | {page_label}]"

        if chunk_type == "image":
            caption = p.get("text", "")
            context_parts.append(f"{label} [Image]\nCaption: {caption}")
            content_blocks.append({"type": "text", "text": f"{label} [Image]\n"})
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": p.get("image_media_type", "image/jpeg"),
                    "data": p["image_base64"],
                },
            })
            b64 = p.get("image_base64", "")
            sources.append({
                "rank": rank_i + 1,
                "filename": p["filename"],
                "page": p["page"],
                "doc_id": p["doc_id"],
                "similarity": round(float(hit.score), 4),
                "snippet": caption[:220] + ("…" if len(caption) > 220 else ""),
                "chunk_type": "image",
                "image_base64": b64 if len(b64) <= 200_000 else "",
                "image_media_type": p.get("image_media_type", "image/jpeg"),
            })
        else:
            doc_text = p["text"]
            context_parts.append(f"{label}\n{doc_text}")
            content_blocks.append({"type": "text", "text": f"{label}\n{doc_text}"})
            sources.append({
                "rank": rank_i + 1,
                "filename": p["filename"],
                "page": p["page"],
                "doc_id": p["doc_id"],
                "similarity": round(float(hit.score), 4),
                "snippet": doc_text[:220] + ("…" if len(doc_text) > 220 else ""),
                "chunk_type": "text",
            })

    return "\n\n---\n\n".join(context_parts), content_blocks, sources


async def rag_stream(query: str, doc_ids: list[str] | None) -> AsyncIterator[str]:
    q_vec = embed([query])[0]
    results = _search(q_vec, doc_ids, TOP_K_RETRIEVE)

    if not results:
        yield "data: " + json.dumps({"type": "error", "content": "No relevant content found."}) + "\n\n"
        return

    context, content_blocks, sources = _build_context(_rerank(query, _hybrid_fuse(query, results)))

    yield "data: " + json.dumps({"type": "sources", "sources": sources}) + "\n\n"

    has_images = any(b.get("type") == "image" for b in content_blocks)
    if has_images:
        user_content = (
            [{"type": "text", "text": "Context from uploaded documents:\n\n"}]
            + content_blocks
            + [{"type": "text", "text": f"\n---\n\nQuestion: {query}\n\nAnswer based on the context above, citing [Source N, Page X] for each claim. When an image is relevant, describe what you see in it."}]
        )
    else:
        user_content = (
            f"Context from uploaded documents:\n\n{context}\n\n"
            f"---\n\nQuestion: {query}\n\n"
            "Answer based on the context above, citing [Source N, Page X] for each claim:"
        )

    with state.anthropic_client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=CHAT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        for text_chunk in stream.text_stream:
            yield "data: " + json.dumps({"type": "token", "content": text_chunk}) + "\n\n"

    yield "data: " + json.dumps({"type": "done"}) + "\n\n"


def rag_answer(query: str, doc_ids: list[str] | None = None) -> dict:
    q_vec = embed([query])[0]
    results = _search(q_vec, doc_ids, TOP_K_RETRIEVE)

    if not results:
        return {"answer": "No relevant content found.", "context": "", "retrieved_count": 0}

    context, _, _ = _build_context(_rerank(query, _hybrid_fuse(query, results)))
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
