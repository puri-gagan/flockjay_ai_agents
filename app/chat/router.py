"""POST /chat — streams the agent's text reply as plain text.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.genai.types import Content, Part

from app.agent.reflect_retry import build_plugin as build_reflect_retry_plugin
from app.agent.root_agent import build_root_agent
from app.attachments.ingest import AttachmentError, register_attachment
from app.attachments.registry import registry
from app.auth import require_api_key
from app.chat.schemas import Attachment, ChatRequest
from app.constants import DEFAULT_USER_ID
from app.runtime.runner import get_runtime

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", dependencies=[Depends(require_api_key)])
async def chat(req: ChatRequest) -> StreamingResponse:
    runtime = get_runtime()
    session_id = req.session_id or str(uuid.uuid4())
    user_id = req.user_id or DEFAULT_USER_ID
    
    # log message preview
    preview = req.message[:200] + ("…" if len(req.message) > 200 else "")
    log.info(
        "running agent with user input: session=%s user=%s attachments=%d input=%r",
        session_id, user_id, len(req.attachments or []), preview,
    )

    # 1. Ingest attachments, collect error on loading
    attachment_errors = await _ingest_new_attachments(
        session_id=session_id,
        attachments=req.attachments or [],
        runtime=runtime,
    )

    # 2. Build the agent with Opik callbacks and session note carrying attachment context goes into the instruction.
    session_note = registry.build_session_note(session_id)
    agent = build_root_agent(session_note=session_note)

    return StreamingResponse(
        _stream_agent_reply(
            agent=agent,
            runtime=runtime,
            user_id=user_id,
            session_id=session_id,
            user_message=req.message,
            attachment_errors=attachment_errors,
        ),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Session-Id": session_id,
            "X-Accel-Buffering": "no",  # disables nginx response buffering
            "Cache-Control": "no-cache",
        },
    )

async def _stream_agent_reply(
    *,
    agent,
    runtime,
    user_id: str,
    session_id: str,
    user_message: str,
    attachment_errors: list[str],
) -> AsyncIterator[bytes]:
    """Yield the agent's reply as raw text chunks.

    Runner context manager handles closing the agent's toolsets (MCP HTTP session, etc.)
    """
    new_message = Content(role="user", parts=[Part.from_text(text=user_message)])

    async with Runner(
        app_name="flockjay",
        agent=agent,
        session_service=runtime.session_service,
        artifact_service=runtime.artifact_service,
        plugins=[build_reflect_retry_plugin(max_retries=2)],
        auto_create_session=True,
    ) as runner:
        try:
            for err in attachment_errors:
                yield f"[attachment error: {err}]\n\n".encode("utf-8")

            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=new_message,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            ):
                for fc in event.get_function_calls() or []:
                    log.debug("tool call → %s args=%s", fc.name, fc.args)
                for fr in event.get_function_responses() or []:
                    log.debug("tool response ← %s", fr.name)

                if not event.partial:
                    continue
                if not event.content or not event.content.parts:
                    continue
                for part in event.content.parts:
                    if part.text:
                        yield part.text.encode("utf-8")
        except Exception as exc:
            log.exception("Agent stream failed")
            # Headers are already on the wire; communicate the error inline.
            yield f"\n\n[stream error: {exc.__class__.__name__}: {exc}]".encode("utf-8")


async def _ingest_new_attachments(
    *,
    session_id: str,
    attachments: list[Attachment],
    runtime,
) -> list[str]:
    """Ingest attachments; return human-readable error strings for any failures."""
    if not attachments:
        return []

    log.info("ingesting %d attachment(s) for session=%s", len(attachments), session_id)

    async def _one(att: Attachment) -> str | None:
        try:
            await register_attachment(
                session_id=session_id,
                attachment_id=att.id,
                url=att.url,
                type_hint=att.type,
                embedder=runtime.embedder,
                summarizer=runtime.summarizer,
            )
            return None
        except AttachmentError as exc:
            log.warning("Attachment %s failed: %s", att.id, exc)
            return f"{att.id}: {exc}"
        except Exception as exc:
            log.exception("Unexpected attachment error for %s", att.id)
            return f"{att.id}: {exc.__class__.__name__}: {exc}"

    results = await asyncio.gather(*(_one(a) for a in attachments))
    return [r for r in results if r]


