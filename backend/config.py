import os
from pathlib import Path

UPLOAD_DIR = Path("/app/uploads")

QDRANT_URL     = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
COLLECTION_NAME = "documents"

CHUNK_SIZE_TOKENS    = 400   # tokens — SentenceSplitter unit
CHUNK_OVERLAP_TOKENS = 50    # tokens
MIN_CHUNK_CHARS      = 100   # discard nodes shorter than this

TOP_K_RETRIEVE = 15
TOP_K_RERANK   = 5
