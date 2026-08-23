import re
import uuid
from pathlib import Path

import pypdf
try:
    import pymupdf as fitz
except ImportError:
    import fitz

from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, MIN_CHUNK_CHARS

MIN_IMAGE_BYTES = 5_000   # skip tiny icons/decorations
MAX_IMAGE_BYTES = 400_000  # skip extremely large images that would bloat SSE


def _tokens(text: str) -> int:
    # words * 1.3 ≈ tokens — good enough for chunking
    return int(len(text.split()) * 1.3)


def _sentences(text: str) -> list[str]:
    # split on sentence-ending punctuation or paragraph breaks
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\d])|(?:\n\n+)', text)
    return [p.strip() for p in parts if p.strip()]


def pdf_to_chunks(pdf_path: Path, doc_id: str, filename: str) -> list[dict]:
    reader = pypdf.PdfReader(str(pdf_path))
    chunks = []
    chunk_index = 0

    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        sents = _sentences(text)
        current: list[str] = []
        current_tok = 0

        for sent in sents:
            s_tok = _tokens(sent)

            if current_tok + s_tok > CHUNK_SIZE_TOKENS and current:
                body = " ".join(current)
                if len(body) >= MIN_CHUNK_CHARS:
                    chunks.append({
                        "text": body,
                        "doc_id": doc_id,
                        "filename": filename,
                        "page": page_num,
                        "chunk_index": chunk_index,
                    })
                    chunk_index += 1

                # carry overlap sentences forward
                overlap, overlap_tok = [], 0
                for s in reversed(current):
                    t = _tokens(s)
                    if overlap_tok + t <= CHUNK_OVERLAP_TOKENS:
                        overlap.insert(0, s)
                        overlap_tok += t
                    else:
                        break
                current, current_tok = overlap, overlap_tok

            current.append(sent)
            current_tok += s_tok

        if current:
            body = " ".join(current)
            if len(body) >= MIN_CHUNK_CHARS:
                chunks.append({
                    "text": body,
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": page_num,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

    return chunks


def chunk_id(doc_id: str, page: int, chunk_idx: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}::p{page}::c{chunk_idx}"))


def image_chunk_id(doc_id: str, page: int, img_idx: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}::img::p{page}::i{img_idx}"))


def pdf_to_image_chunks(pdf_path: Path, doc_id: str, filename: str) -> list[dict]:
    """Extract embedded raster images from a PDF using pymupdf."""
    doc = fitz.open(str(pdf_path))
    chunks = []
    img_index = 0

    for page_num, page in enumerate(doc, start=1):
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:  # CMYK → convert to RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("jpeg", jpg_quality=75)
            except Exception:
                continue

            if len(img_bytes) < MIN_IMAGE_BYTES or len(img_bytes) > MAX_IMAGE_BYTES:
                continue

            chunks.append({
                "type": "image",
                "doc_id": doc_id,
                "filename": filename,
                "page": page_num,
                "img_index": img_index,
                "image_bytes": img_bytes,
                "image_media_type": "image/jpeg",
            })
            img_index += 1

    doc.close()
    return chunks
