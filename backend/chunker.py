"""
Adaptive multi-strategy PDF chunker.

Strategy is auto-detected per page based on content heuristics:

  prose    — research papers, continuous narrative text
             Uses the custom paragraph-aware chunker (best for academic docs).

  sections — manuals, technical docs with visible section headings
             Uses LangChain RecursiveCharacterTextSplitter within each detected
             section, and prepends the heading to every chunk so retrieval
             always returns the heading alongside its content.

Tables are extracted separately by pdfplumber (before either strategy runs)
and formatted as pipe-delimited text so numeric content stays intact.
The synthetic page 0 (first 3 pages combined) is always created for
front-matter / author / title retrieval.
"""

import re
from pathlib import Path

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import MIN_CHUNK_SIZE, MAX_CHUNK_SIZE, CHUNK_OVERLAP

# LangChain splitter used inside section-based chunking
_lc_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=MAX_CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)


# ── Text cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n([a-z])", r"\1", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ── Table formatting ─────────────────────────────────────────────────────────

def _format_table(table: list[list]) -> str:
    """Convert a pdfplumber table to pipe-delimited text rows."""
    rows = []
    for row in table:
        rows.append(" | ".join(str(cell or "").strip() for cell in row))
    return "\n".join(rows)


# ── Strategy detection ───────────────────────────────────────────────────────

def _is_heading(line: str) -> bool:
    """
    Heuristic: a heading is short, title-like, and doesn't end with a period.
    Catches: 'Introduction', '1.2 Methods', 'APPENDIX A', 'Chapter 3'
    """
    line = line.strip()
    if not (3 < len(line) < 80):
        return False
    if line.endswith((".", ",")):
        return False
    return (
        line.istitle()
        or line.isupper()
        or bool(re.match(r"^\d+[\.\d]*\s+[A-Z]", line))   # e.g. "1.2 Introduction"
        or bool(re.match(r"^(Chapter|Section|Appendix)\s", line, re.I))
    )


def _detect_strategy(text: str) -> str:
    """Return 'sections' for manuals/tech docs, 'prose' for everything else."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return "prose"
    heading_ratio = sum(1 for l in lines if _is_heading(l)) / len(lines)
    return "sections" if heading_ratio > 0.08 else "prose"


# ── Strategy 1: Custom prose chunker (research papers) ───────────────────────

def _sliding_window(text: str) -> list[str]:
    """Fallback within prose chunking: character-level window with sentence snapping."""
    chunks, start = [], 0
    while start < len(text):
        end = start + MAX_CHUNK_SIZE
        chunk = text[start:end]
        if end < len(text):
            last_break = max(chunk.rfind(". "), chunk.rfind("\n"))
            if last_break > MAX_CHUNK_SIZE // 2:
                end = start + last_break + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        next_start = end - CHUNK_OVERLAP
        while next_start < len(text) and text[next_start] not in (" ", "\n", "."):
            next_start += 1
        start = next_start + 1 if next_start < len(text) else len(text)
    return [c for c in chunks if len(c) >= MIN_CHUNK_SIZE]


def paragraph_chunk(text: str) -> list[str]:
    """
    Custom paragraph-aware chunker — best for research papers and dense prose.
    Merges short paragraphs, splits oversized ones with a sliding window.
    Kept as the default prose strategy because it outperforms generic splitters
    on academic documents.
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buffer = [], ""
    for para in raw_paragraphs:
        if len(buffer) + len(para) + 2 <= MAX_CHUNK_SIZE:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para
        else:
            if len(buffer) >= MIN_CHUNK_SIZE:
                chunks.append(buffer)
            if len(para) > MAX_CHUNK_SIZE:
                chunks.extend(_sliding_window(para))
                buffer = ""
            else:
                buffer = para
    if len(buffer) >= MIN_CHUNK_SIZE:
        chunks.append(buffer)
    return chunks


# ── Strategy 2: LangChain section-aware chunker (manuals / technical docs) ───

def _chunk_sections(text: str) -> list[str]:
    """
    Section-aware chunking for manuals and technical documents.
    - Splits text at detected headings.
    - Prepends the current section heading to every chunk so that retrieval
      always returns the heading alongside its content.
    - Uses LangChain RecursiveCharacterTextSplitter within each section body
      to handle sections of any length cleanly.
    - Falls back to paragraph_chunk if no headings are detected.
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading, current_body = "", []

    for line in lines:
        if _is_heading(line):
            if current_body:
                sections.append((current_heading, current_body))
            current_heading = line.strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_heading, current_body))

    if not sections:
        return paragraph_chunk(text)  # no headings found — fall back to prose

    chunks = []
    for heading, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        prefix = f"{heading}\n\n" if heading else ""
        for chunk in _lc_splitter.split_text(body):
            if len(chunk) >= MIN_CHUNK_SIZE:
                chunks.append((prefix + chunk).strip())

    return chunks if chunks else paragraph_chunk(text)


# ── Public entry point ───────────────────────────────────────────────────────

def smart_chunk(text: str) -> list[str]:
    """
    Route to the right strategy based on content analysis.
    - Manuals / technical docs with headings → section-aware (LangChain)
    - Research papers / continuous prose    → paragraph-aware (custom)
    Tables have already been extracted and appended by pdf_to_pages before
    this function is called.
    """
    if _detect_strategy(text) == "sections":
        return _chunk_sections(text)
    return paragraph_chunk(text)


# ── PDF extraction ───────────────────────────────────────────────────────────

def pdf_to_pages(pdf_path: Path) -> list[dict]:
    """
    Extract text and tables per page using pdfplumber.
    - Tables are formatted as pipe-delimited text and appended to page text
      so numeric/tabular content is preserved and retrievable.
    - Synthetic page 0 concatenates the first 3 pages for front-matter retrieval
      (author, title, abstract).
    """
    pages = []
    first_pages_text = ""

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            # Extract and format any tables on this page
            raw_tables = page.extract_tables() or []
            table_blocks = [_format_table(t) for t in raw_tables if t]

            # Extract main body text
            text = clean_text(page.extract_text() or "")

            # Merge table text into the page (tables get their own paragraphs)
            if table_blocks:
                text = (text + "\n\n" + "\n\n".join(table_blocks)).strip()

            if text:
                pages.append({"page": i + 1, "text": text})
                if i < 3:
                    first_pages_text += text + "\n\n"

    # Synthetic front-matter chunk
    if first_pages_text.strip():
        pages.insert(0, {"page": 0, "text": first_pages_text.strip()})

    return pages


def chunk_id(doc_id: str, page: int, chunk_idx: int) -> str:
    return f"{doc_id}::p{page}::c{chunk_idx}"
