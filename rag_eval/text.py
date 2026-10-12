"""PDF loading and Japanese-aware chunking that keeps page numbers."""

import re
import unicodedata
from pathlib import Path
from typing import Dict, List

_WS = re.compile(r"\s+")
_SENT = re.compile(r"(?<=[。！？])")


def normalize(text: str) -> str:
    """NFKC + drop all whitespace (PDF line breaks split Japanese words)."""
    return _WS.sub("", unicodedata.normalize("NFKC", text))


def load_pdf_pages(path: Path) -> List[Dict]:
    import pymupdf  # imported lazily so unit tests need no PDF library

    doc = pymupdf.open(path)
    return [
        {"doc": Path(path).name, "page": i + 1, "text": normalize(p.get_text())}
        for i, p in enumerate(doc)
    ]


def chunk_page(page: Dict, size: int = 512, overlap: int = 50) -> List[Dict]:
    """Split one page into chunks of <= size chars on sentence boundaries, with char overlap."""
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    sentences = [s for s in _SENT.split(page["text"]) if s]
    pieces: List[str] = []
    for s in sentences:  # hard-split very long "sentences" (tables, figure text)
        while len(s) > size:
            pieces.append(s[:size])
            s = s[size - overlap:]
        if s:
            pieces.append(s)
    chunks, cur = [], ""
    for s in pieces:
        if cur and len(cur) + len(s) > size:
            chunks.append(cur)
            keep = min(overlap, size - len(s))
            cur = cur[-keep:] if keep > 0 else ""
        cur += s
    if cur:
        chunks.append(cur)
    return [
        {"doc": page["doc"], "page": page["page"], "chunk_id": f'{page["doc"]}#p{page["page"]}#{i}', "text": c}
        for i, c in enumerate(chunks)
    ]


def chunk_pages(pages: List[Dict], size: int = 512, overlap: int = 50) -> List[Dict]:
    return [c for p in pages for c in chunk_page(p, size, overlap)]
