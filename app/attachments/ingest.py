"""Attachment ingestion pipeline.

Given a (session_id, attachment_spec), download the content to a temp file,
extract plain text, chunk it with tiktoken, embed via Gemini, summarize via
Gemini, and register all of it in the module-level AttachmentRegistry.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
import tiktoken

from app.attachments.embedder import GeminiEmbedder
from app.attachments.extractors import extract_text
from app.attachments.registry import AttachmentRecord, registry
from app.attachments.summarizer import Summarizer
from app.constants import (
    ATTACHMENT_DOWNLOAD_CONNECT_TIMEOUT_SECONDS,
    ATTACHMENT_DOWNLOAD_TIMEOUT_SECONDS,
    ATTACHMENT_MAX_BYTES,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    TIKTOKEN_ENCODING,
)

log = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT = httpx.Timeout(
    ATTACHMENT_DOWNLOAD_TIMEOUT_SECONDS,
    connect=ATTACHMENT_DOWNLOAD_CONNECT_TIMEOUT_SECONDS,
)


class AttachmentError(Exception):
    """Raised when an attachment cannot be ingested."""


async def register_attachment(
    *,
    session_id: str,
    attachment_id: str,
    url: str,
    type_hint: str | None,
    embedder: GeminiEmbedder,
    summarizer: Summarizer,
) -> AttachmentRecord:
    """Download, extract, chunk, embed, summarize, and register an attachment.

    Idempotent: re-registering the same (session_id, attachment_id) returns
    the existing record without re-doing the work.
    """
    existing = registry.get(session_id, attachment_id)
    if existing is not None:
        log.debug("attachment already registered: %s/%s", session_id, attachment_id)
        return existing

    log.info(
        "attachment %s: starting ingestion (session=%s url=%s type_hint=%s)",
        attachment_id, session_id, url, type_hint,
    )

    async with _download(url) as (path, content_type):
        log.debug(
            "attachment %s: downloaded (content_type=%s size=%d bytes)",
            attachment_id, content_type, path.stat().st_size,
        )
        text = await asyncio.to_thread(extract_text, path, content_type, type_hint)

    if not text.strip():
        raise AttachmentError(f"attachment {attachment_id} had no extractable text")

    chunks = _chunk(text, CHUNK_SIZE, CHUNK_OVERLAP)
    log.info(
        "attachment %s: extracted %d chars -> %d chunks", attachment_id, len(text), len(chunks)
    )

    # Embed + summarize in parallel. Embedder failure aborts (no embeddings = no
    # semantic search), but summarizer failure is non-fatal — the attachment is
    # still useful via search_attachment, just without a summary card.
    embeddings_task = asyncio.create_task(embedder.embed(chunks))
    summary_task = asyncio.create_task(summarizer.summarize(text))

    embeddings = await embeddings_task
    try:
        summary = await summary_task
    except Exception as exc:
        log.warning(
            "attachment %s: summarizer failed (%s: %s); proceeding without summary",
            attachment_id, exc.__class__.__name__, exc,
        )
        summary = ""

    record = registry.register(
        session_id=session_id,
        attachment_id=attachment_id,
        source_url=url,
        type=(type_hint or _infer_type(content_type, Path(url).suffix)),
        chunks=chunks,
        embeddings=embeddings,
        summary=summary,
    )
    log.info(
        "attachment %s: ingested OK (type=%s chunks=%d)",
        attachment_id, record.type, len(chunks),
    )
    return record


@asynccontextmanager
async def _download(url: str) -> AsyncIterator[tuple[Path, str | None]]:
    """Stream-download a URL into a tempfile; yield the path and content-type."""
    suffix = Path(url.split("?", 1)[0]).suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    path = Path(tmp.name)

    try:
        async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise AttachmentError(
                        f"failed to download {url}: HTTP {resp.status_code}"
                    )
                total = 0
                with path.open("wb") as fh:
                    async for chunk in resp.aiter_bytes():
                        total += len(chunk)
                        if total > ATTACHMENT_MAX_BYTES:
                            raise AttachmentError(
                                f"attachment exceeds {ATTACHMENT_MAX_BYTES // (1024 * 1024)}MB limit"
                            )
                        fh.write(chunk)
                content_type = resp.headers.get("content-type")
        yield path, content_type
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


ENCODER = tiktoken.get_encoding(TIKTOKEN_ENCODING)


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    """Token-aware chunking with overlap. Falls back to raw text if too short."""
    tokens = ENCODER.encode(text)
    if len(tokens) <= size:
        return [text]

    out: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(tokens), step):
        window = tokens[start : start + size]
        if not window:
            break
        out.append(ENCODER.decode(window))
        if start + size >= len(tokens):
            break
    return out


def _infer_type(content_type: str | None, ext: str) -> str:
    ct = (content_type or "").lower()
    if "pdf" in ct or ext.lower() == ".pdf":
        return "pdf"
    if "json" in ct or ext.lower() == ".json":
        return "transcript"
    if ct.startswith("text/"):
        return "text"
    return "unknown"
