import os
from pathlib import Path

UPLOAD_DIR = Path("/app/uploads")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
COLLECTION_NAME = "documents"

MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 1500
CHUNK_OVERLAP  = 200
TOP_K_RETRIEVE = 15
TOP_K_RERANK   = 5
