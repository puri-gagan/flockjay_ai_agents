"""Process-wide runtime singletons.

Instantiated once in the FastAPI lifespan and imported by the chat router.
Keeps heavy SDK clients (the embedder, in particular) out of per-request
hot paths.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.artifacts import InMemoryArtifactService
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool

from app.attachments.embedder import Embedder, build_embedder
from app.attachments.registry import AttachmentRegistry
from app.attachments.tools import build_attachment_tools
from app.constants import (
    COMPACTION_EVENT_RETENTION_SIZE,
    COMPACTION_INTERVAL,
    COMPACTION_OVERLAP_SIZE,
    COMPACTION_SUMMARIZER_MODEL,
    COMPACTION_TOKEN_THRESHOLD,
    EMBEDDING_MODEL,
    EMBEDDING_OUTPUT_DIM,
    LLM_MODEL,
)
from app.settings import settings

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    session_service: InMemorySessionService
    artifact_service: InMemoryArtifactService
    embedder: Embedder
    attachment_registry: AttachmentRegistry
    attachment_tools: list[FunctionTool]
    compaction_config: EventsCompactionConfig

    async def aclose(self) -> None:
        # SDKs (google-genai, openai) manage their own httpx pools internally;
        # no explicit close needed.
        return


_runtime: Runtime | None = None


def init_runtime() -> Runtime:
    """Build and bind the runtime. Idempotent — returns the existing one on re-init."""
    global _runtime
    if _runtime is not None:
        return _runtime

    _normalize_provider_env()

    embedder = build_embedder(EMBEDDING_MODEL, EMBEDDING_OUTPUT_DIM)
    attachment_registry = AttachmentRegistry()
    attachment_tools = build_attachment_tools(
        embedder=embedder, registry=attachment_registry
    )

    compaction_config = EventsCompactionConfig(
        compaction_interval=COMPACTION_INTERVAL,
        overlap_size=COMPACTION_OVERLAP_SIZE,
        token_threshold=COMPACTION_TOKEN_THRESHOLD,
        event_retention_size=COMPACTION_EVENT_RETENTION_SIZE,
        summarizer=LlmEventSummarizer(llm=Gemini(model=COMPACTION_SUMMARIZER_MODEL)),
    )

    _runtime = Runtime(
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
        embedder=embedder,
        attachment_registry=attachment_registry,
        attachment_tools=attachment_tools,
        compaction_config=compaction_config,
    )

    log.info(
        "Runtime initialized (llm=%s, embedding=%s, dim=%d, "
        "compaction_token_threshold=%d, retention=%d, summarizer=%s)",
        LLM_MODEL, EMBEDDING_MODEL, EMBEDDING_OUTPUT_DIM,
        COMPACTION_TOKEN_THRESHOLD, COMPACTION_EVENT_RETENTION_SIZE,
        COMPACTION_SUMMARIZER_MODEL,
    )
    return _runtime


def get_runtime() -> Runtime:
    if _runtime is None:
        raise RuntimeError("Runtime not initialized — call init_runtime() from lifespan")
    return _runtime


# ----------------------------------------------------------------------


def _normalize_provider_env() -> None:
    """Export provider keys to env vars in the form each SDK expects.

    * Gemini: ADK + LiteLLM both look at ``GOOGLE_API_KEY``. Pop
      ``GEMINI_API_KEY`` if both are set so google-genai stops emitting
      the "Both ... are set" warning on every client construction.
    * OpenAI: SDK and LiteLLM read ``OPENAI_API_KEY`` directly.
    * Anthropic: LiteLLM reads ``ANTHROPIC_API_KEY`` directly.

    Without this step, keys living in ``.env`` would be visible to
    ``settings`` but invisible to the SDKs that read raw env vars.
    """
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
        os.environ.pop("GEMINI_API_KEY", None)

    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    if settings.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
