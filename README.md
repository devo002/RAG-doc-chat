# RAG Document Chat

A full-stack web application that lets you upload PDFs and chat with them using Retrieval-Augmented Generation (RAG). Built with **FastAPI**, **React + Vite**, **Qdrant Cloud**, and **Anthropic Claude**.

---

## Architecture

```
┌───────────────┐     HTTP / SSE     ┌──────────────────────┐     HTTPS    ┌─────────────────┐
│  React + Vite  │ ←───────────────→ │  FastAPI Backend      │ ←──────────→ │  Qdrant Cloud   │
│  (port 3000)   │                   │  (port 8000)          │              │  (vector DB)    │
└───────────────┘                    │  · PDF parsing        │              └─────────────────┘
                                     │  · LlamaIndex chunks  │
                                     │  · ONNX embeddings    │     HTTPS    ┌─────────────────┐
                                     │  · Streaming chat     │ ←──────────→ │  Anthropic      │
                                     │  · RAGAS evaluation   │              │  Claude API     │
                                     └──────────────────────┘              └─────────────────┘
```

---

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- An Anthropic API key ([get one here](https://console.anthropic.com))
- A Qdrant Cloud account with a free cluster ([get one here](https://cloud.qdrant.io))

### Steps

```bash
# 1. Clone the project
cd rag-doc-chat

# 2. Create your .env file
cp .env.example .env
# Then fill in the three required keys (see below)

# 3. Build and start all services
docker compose up --build

# 4. Open the app
open http://localhost:3000
```

### Environment Variables

Create a `.env` file at the project root with these three values:

```env
ANTHROPIC_API_KEY=sk-ant-...
QDRANT_URL=https://xxxx.eu-central-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# Uncomment and set this only when deploying to a remote server
# VITE_API_URL=https://your-domain.com
```

**Getting your Qdrant credentials:**
1. Go to [cloud.qdrant.io](https://cloud.qdrant.io) → create a free cluster
2. Copy the **Cluster URL** from the cluster overview page
3. Go to **Access Management** → **API Keys** → create a key and copy it

> First build takes a few minutes as it downloads the ONNX embedding model. Subsequent starts are fast because the model is cached in the Docker image.

---

## Using the App

### 1. Upload a PDF
Drag and drop a PDF into the upload area on the left sidebar, or click to browse. A spinner shows while the file is being processed and indexed. The document appears in the list once ready.

### 2. Chat with your document
Type a question and press Enter or click Send. The answer streams in real time with inline citations showing the source file and page number, e.g. `[Source 1, Page 5]`. Click **5 sources retrieved** below any answer to expand and see exactly which pages were used.

### 3. Filter by document
Click a document name in the sidebar to filter the chat to that document only. Click again to deselect.

### 4. Run the RAG Evaluation
Click **Run RAG Benchmark** at the bottom of the sidebar. It streams 10 questions one by one against the bundled benchmark document (*Attention Is All You Need*) and scores each answer live using two RAGAS metrics — **Faithfulness** and **Answer Relevancy**. Results are independent of any documents you have uploaded.

---

## Chunking Strategy

PDFs are parsed and chunked using **LlamaIndex**:

- **PDF loading** — `PDFReader` (pypdf) extracts text page by page, preserving page number metadata
- **Chunking** — `SentenceSplitter` splits each page into nodes of ~400 tokens with 50-token overlap, snapping cuts to sentence boundaries
- **Filtering** — nodes shorter than 100 characters are discarded

This approach keeps sentences intact, avoids mid-word cuts, and produces consistently sized chunks that embed well.

---

## Evaluation — RAGAS-Inspired Scoring

The `/evaluate` endpoint benchmarks the full RAG pipeline against a fixed document so results are always comparable. The benchmark document — **"Attention Is All You Need" (Vaswani et al., 2017)** — is bundled with the backend and indexed automatically on startup.

**10 domain-specific questions** cover concrete facts from the paper: number of attention heads, model dimensions, encoder/decoder layer counts, WMT translation BLEU scores, optimizer hyperparameters, dropout rate, training time, and the purpose of positional encoding.

Ground truth answers for all 10 questions are stored in `backend/eval_data.json` and used by the **Correctness** metric.

Each answer is scored by **Claude acting as an LLM judge** across three metrics in a single call:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Are all claims in the answer supported by the retrieved chunks? |
| **Answer Relevancy** | Does the answer actually address the question asked? |
| **Correctness** | Does the answer match the ground truth stored in `eval_data.json`? |
| **RAGAS Score** | Average of all three above |

The UI shows each question's generated answer alongside the reference ground truth so you can spot where the pipeline goes wrong. Results stream in one question at a time. Scores can vary slightly between runs because vector retrieval and LLM reranking are non-deterministic — a borderline chunk may rank in or out, which shifts the answer.

> **Judge bias note:** Claude Sonnet generates the answers; Claude Haiku (temperature = 0) acts as the judge. Using a smaller, separate model with fixed temperature reduces self-grading leniency.

### Sample Results — 3-Run Average

Scores below are averaged across 3 independent runs on the bundled benchmark document, after adding hybrid BM25 + dense retrieval with Reciprocal Rank Fusion.

| # | Question | Faith | Relevancy | Correct | RAGAS |
|---|---|:---:|:---:|:---:|:---:|
| Q1 | Attention heads in multi-head attention | 100% | 100% | 100% | 100% |
| Q2 | d_model of the base Transformer | 100% | 100% | 100% | 100% |
| Q3 | Encoder / decoder layer count | 100% | 100% | 100% | 100% |
| Q4 | EN-DE BLEU score | 100% | 100% | 100% | 100% |
| Q5 | EN-FR BLEU score | 85% | 100% | 50% | 78% |
| Q6 | Optimizer and beta values | 100% | 100% | 100% | 100% |
| Q7 | Feed-forward inner dimension (d_ff) | 100% | 100% | 100% | 100% |
| Q8 | Dropout rate | 97% | 100% | 100% | 99% |
| Q9 | Training time and hardware | 100% | 100% | 100% | 100% |
| Q10 | Purpose of positional encoding | 98% | 98% | 97% | 98% |
| **Avg** | | **98%** | **100%** | **95%** | **98%** |

**Remaining failure and why:**

- **Q5 (EN-FR BLEU, 50% correctness)** — The paper contains two different figures: **41.8** in the abstract and Table 2 (big model), and **41.0** in the Section 6 prose. Both chunks are retrieved, creating conflicting signals. The pipeline returns 41.8 (which is correct per the abstract), but the judge sees the mixed context and scores conservatively. This is a genuine document-level ambiguity, not a pipeline bug.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite |
| Backend | FastAPI + uvicorn |
| Vector DB | Qdrant Cloud (free tier, managed) |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (ONNX) |
| LLM | Anthropic Claude `claude-sonnet-4-6` |
| PDF parsing | LlamaIndex `PDFReader` (pypdf) |
| Chunking | LlamaIndex `SentenceSplitter` |
| Containerization | Docker Compose |
| Reverse proxy | nginx (with 50 MB upload limit) |

---

## Challenges & How They Were Solved

### 1. Memory crash on large PDF uploads
**Problem:** When uploading a large PDF, the backend tried to embed all chunks at once. A 100-page document could produce 200+ chunks, and holding all those embedding vectors in memory simultaneously caused an out-of-memory crash.

**Fix:** The upload pipeline was changed to process chunks in batches of 20. Each batch is embedded and upserted to Qdrant before the next batch is loaded, keeping memory usage flat regardless of document size (`backend/main.py`).

---

### 2. ChromaDB was too heavy for cloud deployment
**Problem:** ChromaDB runs as a separate container with an in-memory store, consuming significant RAM — a problem on Railway's free tier where memory is limited.

**Fix:** Replaced ChromaDB with **Qdrant Cloud** (managed, free tier). The vector database now lives entirely outside the deployment, removing one container and eliminating the memory overhead entirely.

### 3. Backend OOM crash on Railway at startup
**Problem:** During startup the backend ran 32 warmup embeddings to pre-initialise the ONNX session. On Railway this spiked memory to 2.5 GB, causing a silent OOM kill before the service could accept any requests.

**Fix:** Reduced the warmup to a single sentence (`backend/main.py`). Memory at startup dropped to under 800 MB and the container stopped crashing.

---

### 4. Railway PORT mismatch — "Application failed to respond"
**Problem:** The Dockerfile hardcoded `--port 8000`, but Railway injects its own `$PORT` variable (e.g. 8080) and routes public traffic to that port. The app was listening on 8000 while Railway probed 8080, so every request returned "Application failed to respond".

**Fix:** Changed the Dockerfile CMD to `uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}` and added `PORT=8000` to Railway's backend Variables so both the app and Railway's router agree on the same port (`backend/Dockerfile`).

---

### 5. nginx proxying to backend using wrong Host header
**Problem:** After switching `BACKEND_URL` to the backend's public Railway URL, nginx was forwarding the frontend's hostname as the `Host` header. Railway's load balancer uses the Host header to route traffic, so it couldn't match the request to the backend service and returned 502/504 errors.

**Fix:** Changed `proxy_set_header Host $host` to `proxy_set_header Host $proxy_host` in the nginx config so the Host header matches the backend's public URL (`frontend/nginx.conf`).

---

## Common Commands

```bash
# Start everything (first time or after code changes)
docker compose up --build

# Rebuild only the backend (after backend code changes)
docker compose up --build backend -d

# Rebuild only the frontend (after UI changes)
docker compose up --build frontend -d

# Stop all services
docker compose down

# Stop and remove all uploaded files
docker compose down -v
```
