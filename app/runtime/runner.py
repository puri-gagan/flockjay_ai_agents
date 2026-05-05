"""Process-wide runtime singletons.

Instantiated once in the FastAPI lifespan and imported by the chat router.
Keeps heavy clients (Gemini for embeddings) out of per-request hot paths.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from google import genai
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService

from app.attachments.embedder import GeminiEmbedder
from app.attachments.summarizer import Summarizer
from app.attachments.tools import set_embedder
from app.constants import EMBEDDING_MODEL, LLM_MODEL, SUMMARIZER_MODEL
from app.settings import settings

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    session_service: InMemorySessionService
    artifact_service: InMemoryArtifactService
    genai_client: genai.Client
    embedder: GeminiEmbedder
    summarizer: Summarizer

    async def aclose(self) -> None:
        # google-genai manages its own httpx pools internally; no explicit
        # close needed.
        return


_runtime: Runtime | None = None


def init_runtime() -> Runtime:
    """Build and bind the runtime. Idempotent — returns the existing one on re-init."""
    global _runtime
    if _runtime is not None:
        return _runtime

    _normalize_provider_env()

    # genai.Client picks up GOOGLE_API_KEY (or GEMINI_API_KEY) from env.
    genai_client = genai.Client()
    embedder = GeminiEmbedder(client=genai_client, model=EMBEDDING_MODEL)
    summarizer = Summarizer(model=SUMMARIZER_MODEL)

    _runtime = Runtime(
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
        genai_client=genai_client,
        embedder=embedder,
        summarizer=summarizer,
    )

    set_embedder(embedder)

    log.info(
        "Runtime initialized (model=%s, summarizer=%s, embedding=%s)",
        LLM_MODEL,
        SUMMARIZER_MODEL,
        EMBEDDING_MODEL,
    )
    return _runtime


def get_runtime() -> Runtime:
    if _runtime is None:
        raise RuntimeError("Runtime not initialized — call init_runtime() from lifespan")
    return _runtime


# ----------------------------------------------------------------------


def _normalize_provider_env() -> None:
    """Export provider keys to env vars in the form LiteLLM / ADK expect.

    * Gemini: ADK + LiteLLM both look at ``GOOGLE_API_KEY``. Pop
      ``GEMINI_API_KEY`` if both are set so google-genai stops emitting the
      "Both ... are set" warning on every client construction.
    * Anthropic: LiteLLM reads ``ANTHROPIC_API_KEY`` directly — no extra
      work needed; we just surface the key from settings if the user put it
      in .env.
    """
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
        os.environ.pop("GEMINI_API_KEY", None)

    if settings.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
