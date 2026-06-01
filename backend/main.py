import hashlib

import anthropic
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastembed import TextEmbedding
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    PayloadSchemaType,
)

import state
from config import (
    UPLOAD_DIR, QDRANT_URL, QDRANT_API_KEY,
    ANTHROPIC_API_KEY, COLLECTION_NAME,
)
from chunker import pdf_to_chunks, chunk_id
from retriever import embed, rag_stream
from evaluator import run_evaluation

# ── App setup ──────────────────────────────────
app = FastAPI(title="RAG Document Chat API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_SIZE = 384  # BAAI/bge-small-en-v1.5 output dimension


# ── Startup ─────────────────────────────────────
@app.on_event("startup")
async def startup():
    print("Loading embedding model…")
    state.embedder = TextEmbedding(
        "BAAI/bge-small-en-v1.5",
        providers=["CPUExecutionProvider"],
    )
    _warmup = ["This is a warmup sentence to initialize the ONNX inference session properly."] * 32
    list(state.embedder.embed(_warmup))
    print("Embedding model ready.")

    print("Connecting to Qdrant…")
    state.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    existing = {c.name for c in state.qdrant_client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        state.qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    # Qdrant requires an explicit index on any field used in filters
    state.qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="doc_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    state.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    print("All services ready ✓")


# ── Helpers ─────────────────────────────────────
def _doc_count() -> int:
    return state.qdrant_client.count(collection_name=COLLECTION_NAME).count


def _doc_exists(doc_id: str) -> bool:
    hits, _ = state.qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(hits) > 0


# ── Routes ──────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents():
    all_points, _ = state.qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=["doc_id", "filename"],
        limit=10_000,
    )
    docs = {}
    for point in all_points:
        doc_id = point.payload.get("doc_id", "")
        filename = point.payload.get("filename", "")
        if doc_id and doc_id not in docs:
            docs[doc_id] = filename
    return [{"doc_id": k, "filename": v} for k, v in docs.items()]


@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")

        content = await file.read()
        doc_id = hashlib.sha256(content).hexdigest()[:16]

        if _doc_exists(doc_id):
            results.append({"filename": file.filename, "doc_id": doc_id, "status": "already_indexed"})
            continue

        save_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"
        save_path.write_bytes(content)

        chunks_data = pdf_to_chunks(save_path, doc_id, file.filename)

        if not chunks_data:
            results.append({"filename": file.filename, "doc_id": doc_id, "status": "no_text_extracted"})
            continue

        texts = [c["text"] for c in chunks_data]
        embeddings = embed(texts)

        points = [
            PointStruct(
                id=chunk_id(c["doc_id"], c["page"], c["chunk_index"]),
                vector=emb,
                payload={
                    "doc_id": c["doc_id"],
                    "filename": c["filename"],
                    "page": c["page"],
                    "chunk_index": c["chunk_index"],
                    "text": c["text"],
                    "char_count": len(c["text"]),
                },
            )
            for c, emb in zip(chunks_data, embeddings)
        ]

        for i in range(0, len(points), 100):
            state.qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i:i + 100],
            )

        results.append({
            "filename": file.filename,
            "doc_id": doc_id,
            "status": "indexed",
            "chunks": len(chunks_data),
        })

    return results


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    state.qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
    )
    return {"status": "deleted", "doc_id": doc_id}


class ChatRequest(BaseModel):
    query: str
    doc_ids: list[str] | None = None


@app.post("/chat")
async def chat(req: ChatRequest):
    if _doc_count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Please upload a PDF first.")
    return StreamingResponse(
        rag_stream(req.query, req.doc_ids),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/evaluate")
def evaluate(doc_ids: str | None = None):
    if _doc_count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet.")
    selected = doc_ids.split(",") if doc_ids else None
    return run_evaluation(selected)
