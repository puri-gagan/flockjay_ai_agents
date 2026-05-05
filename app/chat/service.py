"""ChatService — orchestrates the agent run for a single ``/chat`` request.

This is the use-case layer between the HTTP router and the ADK / attachment
machinery. The router is responsible only for binding to FastAPI; the
service owns:

  - Resolving session and user ids
  - Ingesting the request's attachments (parallel, error-tolerant)
  - Building the per-request agent (with session note + attachment tools)
  - Wrapping it in an ADK ``App`` with plugins + compaction config
  - Driving the ``Runner`` async context manager
  - Streaming the final-text events back as plain bytes

Constructed once at lifespan time with the shared ``Runtime``; reused
across requests. Stateless beyond that.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.genai.types import Content, Part

from app.agent.reflect_retry import build_plugin as build_reflect_retry_plugin
from app.agent.root_agent import build_root_agent
from app.attachments.ingest import AttachmentError, register_attachment
from app.chat.schemas import Attachment, ChatRequest
from app.constants import APP_NAME, DEFAULT_USER_ID
from app.runtime.runner import Runtime

log = logging.getLogger(__name__)


class ChatService:
    """Orchestrates the agent execution for `/chat` requests."""

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime

    async def start_stream(self, req: ChatRequest) -> tuple[str, AsyncIterator[bytes]]:
        """Resolve the session id and return a body iterator.

        Attachment ingestion runs **inside** the iterator so the response
        headers (including ``X-Session-Id``) flush as soon as Starlette starts
        consuming the body — the client sees the connection open immediately,
        even when ingest takes seconds. Body bytes start arriving once ingest
        completes.

        The returned iterator must be consumed inside the same async task that
        called this method — it owns the Runner async context.
        """
        session_id = req.session_id or str(uuid.uuid4())
        user_id = req.user_id or DEFAULT_USER_ID

        preview = req.message[:200] + ("…" if len(req.message) > 200 else "")
        log.info(
            "running agent: session=%s user=%s attachments=%d input=%r",
            session_id, user_id, len(req.attachments or []), preview,
        )

        body = self._stream(
            user_id=user_id,
            session_id=session_id,
            message=req.message,
            attachments=req.attachments or [],
        )
        return session_id, body

    # ------------------------------------------------------------------

    async def _stream(
        self,
        *,
        user_id: str,
        session_id: str,
        message: str,
        attachments: list[Attachment],
    ) -> AsyncIterator[bytes]:
        """Yield the agent's reply as raw text chunks.

        The Runner async-context-manager closes the toolset (MCP HTTP, etc.)
        on exit. Compaction runs at the end of each invocation via
        ``app.events_compaction_config``.
        """
        runtime = self._runtime

        # Step 1+2 — ingest attachments inline so headers flush during ingest.
        attachment_errors = await self._ingest_attachments(
            session_id=session_id,
            attachments=attachments,
        )
        for err in attachment_errors:
            yield f"[attachment error: {err}]\n\n".encode()

        # Step 3 — build the agent AFTER ingest so the session note reflects
        # the freshly registered attachments.
        agent = build_root_agent(
            session_note=runtime.attachment_registry.build_session_note(session_id),
            attachment_tools=runtime.attachment_tools,
        )
        app = App(
            name=APP_NAME,
            root_agent=agent,
            plugins=[build_reflect_retry_plugin(max_retries=2)],
            events_compaction_config=runtime.compaction_config,
        )
        new_message = Content(role="user", parts=[Part.from_text(text=message)])

        # Step 4 — drive the Runner.
        async with Runner(
            app=app,
            session_service=runtime.session_service,
            artifact_service=runtime.artifact_service,
            auto_create_session=True,
        ) as runner:
            try:
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
                yield f"\n\n[stream error: {exc.__class__.__name__}: {exc}]".encode()

    async def _ingest_attachments(
        self,
        *,
        session_id: str,
        attachments: list[Attachment],
    ) -> list[str]:
        """Ingest attachments in parallel; return human-readable errors for failures."""
        if not attachments:
            return []

        log.info("ingesting %d attachment(s) for session=%s", len(attachments), session_id)
        runtime = self._runtime

        async def _one(att: Attachment) -> str | None:
            try:
                await register_attachment(
                    session_id=session_id,
                    attachment_id=att.id,
                    url=att.url,
                    type_hint=att.type,
                    embedder=runtime.embedder,
                    registry=runtime.attachment_registry,
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


# ---------------------------------------------------------------------------
# Lifespan-managed singleton — same shape as ``runtime.get_runtime`` so the
# router can pull the service via FastAPI ``Depends(get_chat_service)``.

_chat_service: ChatService | None = None


def init_chat_service(runtime: Runtime) -> ChatService:
    """Build and bind the chat service. Idempotent."""
    global _chat_service
    if _chat_service is not None:
        return _chat_service
    _chat_service = ChatService(runtime)
    return _chat_service


def get_chat_service() -> ChatService:
    if _chat_service is None:
        raise RuntimeError(
            "ChatService not initialized — call init_chat_service() from lifespan"
        )
    return _chat_service
