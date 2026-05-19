import hashlib

import anthropic
import chromadb
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastembed import TextEmbedding
from pydantic import BaseModel

import state
from config import (
    UPLOAD_DIR, CHROMA_HOST, CHROMA_PORT,
    ANTHROPIC_API_KEY, COLLECTION_NAME,
)
from chunker import pdf_to_pages, smart_chunk, chunk_id
from retriever import embed, rag_stream
from evaluator import run_evaluation

# ── App setup ──────────────────────────────────
app = FastAPI(title="RAG Document Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Startup ─────────────────────────────────────
@app.on_event("startup")
async def startup():
    print("Loading embedding model…")
    state.embedder = TextEmbedding(
        "BAAI/bge-small-en-v1.5",
        providers=["CPUExecutionProvider"],
    )
    # Warm up with a realistic batch to fully initialize the ONNX session.
    # A single short string doesn't exercise the same code path as real chunks.
    _warmup = ["This is a warmup sentence to initialize the ONNX inference session properly."] * 32
    list(state.embedder.embed(_warmup))

    print("Connecting to ChromaDB…")
    import time
    for attempt in range(30):
        try:
            state.chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            state.collection = state.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            break
        except Exception:
            if attempt == 29:
                raise
            print(f"ChromaDB not ready (attempt {attempt + 1}/30), retrying in 2s…")
            time.sleep(2)

    state.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    print("All services ready ✓")


# ── Routes ──────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents():
    results = state.collection.get(include=["metadatas"])
    docs = {}
    for meta in results["metadatas"]:
        doc_id, filename = meta.get("doc_id", ""), meta.get("filename", "")
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

        if state.collection.get(where={"doc_id": doc_id}, limit=1)["ids"]:
            results.append({"filename": file.filename, "doc_id": doc_id, "status": "already_indexed"})
            continue

        save_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"
        save_path.write_bytes(content)

        pages = pdf_to_pages(save_path)
        all_chunks, all_ids, all_metas = [], [], []

        for page_data in pages:
            for idx, chunk in enumerate(smart_chunk(page_data["text"])):
                all_chunks.append(chunk)
                all_ids.append(chunk_id(doc_id, page_data["page"], idx))
                all_metas.append({
                    "doc_id": doc_id,
                    "filename": file.filename,
                    "page": page_data["page"],
                    "chunk_index": idx,
                    "char_count": len(chunk),
                })

        if not all_chunks:
            results.append({"filename": file.filename, "doc_id": doc_id, "status": "no_text_extracted"})
            continue

        for i in range(0, len(all_chunks), 100):
            state.collection.upsert(
                ids=all_ids[i:i + 100],
                embeddings=embed(all_chunks[i:i + 100]),
                documents=all_chunks[i:i + 100],
                metadatas=all_metas[i:i + 100],
            )

        results.append({
            "filename": file.filename,
            "doc_id": doc_id,
            "status": "indexed",
            "chunks": len(all_chunks),
            "pages": len(pages),
        })

    return results


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    state.collection.delete(where={"doc_id": doc_id})
    return {"status": "deleted", "doc_id": doc_id}


class ChatRequest(BaseModel):
    query: str
    doc_ids: list[str] | None = None


@app.post("/chat")
async def chat(req: ChatRequest):
    if state.collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Please upload a PDF first.")
    return StreamingResponse(
        rag_stream(req.query, req.doc_ids),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/evaluate")
def evaluate(doc_ids: str | None = None):
    if state.collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet.")
    selected = doc_ids.split(",") if doc_ids else None
    return run_evaluation(selected)
