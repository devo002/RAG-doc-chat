"""
LlamaIndex-based PDF chunker.

Uses PDFReader (pypdf) for text extraction and SentenceSplitter for chunking.
Each node inherits page metadata from its parent Document.
"""

import uuid
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, MIN_CHUNK_CHARS

_reader = PDFReader()
_splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE_TOKENS,
    chunk_overlap=CHUNK_OVERLAP_TOKENS,
)


def pdf_to_chunks(pdf_path: Path, doc_id: str, filename: str) -> list[dict]:
    """
    Load a PDF and return a flat list of chunk dicts ready for embedding.
    Each dict has: text, doc_id, filename, page, chunk_index.
    """
    documents = _reader.load_data(file=pdf_path)
    nodes = _splitter.get_nodes_from_documents(documents)

    chunks = []
    for idx, node in enumerate(nodes):
        text = node.text.strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue
        try:
            page = int(node.metadata.get("page_label", "1"))
        except (ValueError, TypeError):
            page = 1
        chunks.append({
            "text": text,
            "doc_id": doc_id,
            "filename": filename,
            "page": page,
            "chunk_index": idx,
        })
    return chunks


def chunk_id(doc_id: str, page: int, chunk_idx: int) -> str:
    """Deterministic UUID for a chunk — used as the Qdrant point ID."""
    raw = f"{doc_id}::p{page}::c{chunk_idx}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw))
