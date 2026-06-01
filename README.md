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
Click the **Show Eval Scores** button at the bottom of the sidebar. It runs 5 test questions against your indexed documents and scores each answer using two RAGAS metrics — **Faithfulness** and **Answer Relevancy**.

---

## Chunking Strategy

PDFs are parsed and chunked using **LlamaIndex**:

- **PDF loading** — `PDFReader` (pypdf) extracts text page by page, preserving page number metadata
- **Chunking** — `SentenceSplitter` splits each page into nodes of ~400 tokens with 50-token overlap, snapping cuts to sentence boundaries
- **Filtering** — nodes shorter than 100 characters are discarded

This approach keeps sentences intact, avoids mid-word cuts, and produces consistently sized chunks that embed well.

---

## Evaluation — RAGAS-Inspired Scoring

The `/evaluate` endpoint runs **5 hard-coded test questions** against the indexed documents and uses **Claude as an LLM judge** to score two metrics per question:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Answer Relevancy** | Does the answer actually address the question? |
| **RAGAS Score** | Simple average of the two above |

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
