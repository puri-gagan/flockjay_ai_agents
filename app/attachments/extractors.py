"""Extract plain text from attachment bytes.

PDFs go through PyMuPDF to extract the selectable text, 
need to ocr or direct pass to llm with artifact handling and adk artifact service loader (intent based loading) for scanned pdfs
everything else is read as UTF-8 text.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

log = logging.getLogger(__name__)


def extract_text(path: Path, content_type: str | None, hint: str | None = None) -> str:
    """Return the file's text content as-is."""
    ct = (content_type or "").lower().split(";", 1)[0].strip()
    ext = path.suffix.lower()
    hint = (hint or "").lower()

    if ct == "application/pdf" or ext == ".pdf" or hint == "pdf":
        return _extract_pdf(path)

    return path.read_text(encoding="utf-8", errors="replace")


def _extract_pdf(path: Path) -> str:
    doc = fitz.open(path)
    try:
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return "\n\n".join(pages)
