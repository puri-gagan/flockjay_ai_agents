"""FunctionTool wrappers exposed to the root agent.

Docstrings as the descriptions are important, ADK feeds them to the LLM as the tool's description.
"""

from __future__ import annotations

import logging
from typing import Any

from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

from app.attachments.embedder import GeminiEmbedder
from app.attachments.registry import registry
from app.constants import MIN_SIMILARITY, TOP_K

log = logging.getLogger(__name__)

# Populated from app.runtime.runner at startup.
_embedder: GeminiEmbedder | None = None


def set_embedder(embedder: GeminiEmbedder) -> None:
    """Wire the shared embedder into this module (called once from lifespan)."""
    global _embedder
    _embedder = embedder


async def search_attachment(
    attachment_id: str, query: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Semantically search inside a specific attachment registered for this session.

    Use this tool when:
      - The user's question references an attached transcript/document
        (e.g. "what did the prospect say about pricing in my call?")
      - You need a concrete quote or excerpt to cite, not just a summary
      - You want to use the attachment as a retrieval anchor before querying
        other MCP tools (e.g. extract the objection phrasing, then pass it
        to search_content to find playbook guidance)

    Do NOT use this tool for:
      - General library content — use search_content (MCP) for that
      - Questions where the attachment summary already has the answer —
        prefer get_attachment_summary, which is much cheaper

    Args:
        attachment_id: the id from list_active_attachments or the session note.
        query: a focused search phrase (objection text, topic, stakeholder,
            question). Prefer specific phrases over broad topics.

    Returns a dict with `ok`, `hits` (list of {text, score, chunk_idx}) or
    a `reason` string on failure.
    """
    if _embedder is None:
        return {"ok": False, "reason": "embedder not initialized"}

    session_id = tool_context._invocation_context.session.id
    if not session_id:
        return {"ok": False, "reason": "no active session"}

    record = registry.get(session_id, attachment_id)
    if record is None:
        available = [r.attachment_id for r in registry.list_for_session(session_id)]
        return {
            "ok": False,
            "reason": f"no attachment '{attachment_id}' in this session",
            "available_attachment_ids": available,
        }

    try:
        [query_embedding] = await _embedder.embed([query])
    except Exception as exc:
        log.exception("Embedding query failed")
        return {"ok": False, "reason": f"embedding call failed: {exc}"}

    hits = registry.query(session_id, attachment_id, query_embedding, TOP_K)
    filtered = [
        {"text": text, "score": round(score, 3), "chunk_idx": idx}
        for (text, score, idx) in hits
        if score >= MIN_SIMILARITY
    ]
    if not filtered:
        return {
            "ok": False,
            "reason": "no chunks scored above similarity threshold",
            "best_score": round(hits[0][1], 3) if hits else 0.0,
        }
    return {"ok": True, "hits": filtered, "attachment_id": attachment_id}


def get_attachment_summary(attachment_id: str, tool_context: ToolContext) -> dict[str, Any]:
    """Return a plain-text summary of an attachment.

    Use this BEFORE search_attachment when the user's question is broad
    (themes, stakeholders, high-level outcomes). The summary covers the whole
    attachment in a few sentences — often enough on its own.

    Args:
        attachment_id: the id from list_active_attachments or the session note.

    Returns: summary (plain text), chunk_count, length_chars, source_url, type.
    """
    session_id = tool_context._invocation_context.session.id
    record = registry.get(session_id, attachment_id)
    if record is None:
        available = [r.attachment_id for r in registry.list_for_session(session_id)]
        return {
            "ok": False,
            "reason": f"no attachment '{attachment_id}' in this session",
            "available_attachment_ids": available,
        }
    return {
        "ok": True,
        "attachment_id": record.attachment_id,
        "type": record.type,
        "source_url": record.source_url,
        "chunk_count": record.chunk_count,
        "length_chars": record.length_chars,
        "summary": record.summary,
    }


def list_active_attachments(tool_context: ToolContext) -> dict[str, Any]:
    """List the attachments registered for the current session.

    Use this when the user refers to "the call", "the transcript", or "this
    document" without naming an attachment_id.
    """
    session_id = tool_context._invocation_context.session.id
    records = registry.list_for_session(session_id)
    return {
        "ok": True,
        "count": len(records),
        "attachments": [
            {
                "attachment_id": r.attachment_id,
                "type": r.type,
                "chunk_count": r.chunk_count,
                "source_url": r.source_url,
            }
            for r in records
        ],
    }


def build_attachment_tools() -> list[FunctionTool]:
    """Return the list of FunctionTools for the root agent."""
    return [
        FunctionTool(search_attachment),
        FunctionTool(get_attachment_summary),
        FunctionTool(list_active_attachments),
    ]
