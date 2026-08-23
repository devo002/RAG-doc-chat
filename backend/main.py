import base64
import hashlib
from pathlib import Path

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
from chunker import pdf_to_chunks, chunk_id, pdf_to_image_chunks, image_chunk_id
from retriever import embed, rag_stream
from evaluator import run_evaluation_stream

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
    list(state.embedder.embed(["warmup"]))
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
    state.qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="chunk_type",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    state.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Auto-index the bundled benchmark document
    eval_pdf = Path(__file__).parent / "testdoc.pdf"
    if eval_pdf.exists():
        content = eval_pdf.read_bytes()
        doc_id = hashlib.sha256(content).hexdigest()[:16]
        if not _doc_exists(doc_id):
            print("Indexing benchmark document (text)…")
            chunks_data = pdf_to_chunks(eval_pdf, doc_id, "testdoc.pdf")
            for i in range(0, len(chunks_data), 20):
                batch = chunks_data[i:i + 20]
                embeddings = embed([c["text"] for c in batch])
                points = [
                    PointStruct(
                        id=chunk_id(c["doc_id"], c["page"], c["chunk_index"]),
                        vector=emb,
                        payload={"doc_id": c["doc_id"], "filename": c["filename"],
                                 "page": c["page"], "chunk_index": c["chunk_index"],
                                 "text": c["text"], "chunk_type": "text",
                                 "char_count": len(c["text"])},
                    )
                    for c, emb in zip(batch, embeddings)
                ]
                state.qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        if not _image_chunks_exist(doc_id):
            print("Indexing benchmark document (images)…")
            n_imgs = _index_images(eval_pdf, doc_id, "testdoc.pdf")
            print(f"  → {n_imgs} image chunks indexed")
        state.eval_doc_id = doc_id
        print(f"Benchmark document ready (id={doc_id})")

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


def _image_chunks_exist(doc_id: str) -> bool:
    hits, _ = state.qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="chunk_type", match=MatchValue(value="image")),
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(hits) > 0


def _caption_image(image_bytes: bytes) -> str:
    b64 = base64.standard_b64encode(image_bytes).decode()
    response = state.anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": "Describe this image in 1-3 sentences for a document search index. Focus on what it shows: chart axes, diagram labels, equation variables, table columns, or figure content. Be specific."},
            ],
        }],
    )
    return response.content[0].text.strip()


def _index_images(pdf_path: Path, doc_id: str, filename: str) -> int:
    image_chunks = pdf_to_image_chunks(pdf_path, doc_id, filename)
    indexed = 0
    for img in image_chunks:
        try:
            caption = _caption_image(img["image_bytes"])
        except Exception as e:
            print(f"Caption failed for {filename} p{img['page']}: {e}")
            continue
        b64 = base64.standard_b64encode(img["image_bytes"]).decode()
        vec = embed([caption])[0]
        point = PointStruct(
            id=image_chunk_id(doc_id, img["page"], img["img_index"]),
            vector=vec,
            payload={
                "doc_id": doc_id,
                "filename": filename,
                "page": img["page"],
                "chunk_index": img["img_index"],
                "text": caption,
                "chunk_type": "image",
                "image_base64": b64,
                "image_media_type": img["image_media_type"],
                "char_count": len(caption),
            },
        )
        state.qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
        indexed += 1
    return indexed


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

        BATCH = 20
        total = 0
        for i in range(0, len(chunks_data), BATCH):
            batch = chunks_data[i:i + BATCH]
            embeddings = embed([c["text"] for c in batch])
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
                        "chunk_type": "text",
                        "char_count": len(c["text"]),
                    },
                )
                for c, emb in zip(batch, embeddings)
            ]
            state.qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
            total += len(batch)

        n_imgs = _index_images(save_path, doc_id, file.filename)

        results.append({
            "filename": file.filename,
            "doc_id": doc_id,
            "status": "indexed",
            "chunks": total,
            "image_chunks": n_imgs,
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
def evaluate():
    if not state.eval_doc_id:
        raise HTTPException(status_code=503, detail="Benchmark document not available.")
    return StreamingResponse(
        run_evaluation_stream([state.eval_doc_id]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
