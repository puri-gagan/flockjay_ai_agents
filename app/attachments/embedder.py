"""Async batched text-embedding helpers — Gemini and OpenAI.

Both implementations satisfy the :class:`Embedder` protocol:
take an iterable of strings, return a list of vectors. They batch
internally against each provider's tested per-request limit and
L2-normalize every output so the registry's ``1 - dist/2`` cosine-similarity
proxy on Chroma's L2 distance stays correct — Gemini does not pre-normalize
below 3072 dimensions, and OpenAI's ``text-embedding-3-*`` lose unit norm
whenever ``dimensions=`` truncates them.

Use :func:`build_embedder` to construct the embedder for the configured
``EMBEDDING_MODEL``; provider is inferred from the model string.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable, Iterable
from typing import Protocol

from google import genai
from google.genai import types
from openai import AsyncOpenAI

log = logging.getLogger(__name__)


class Embedder(Protocol):
    """Structural interface every embedder implementation satisfies."""

    async def embed(self, texts: Iterable[str]) -> list[list[float]]: ...


class GeminiEmbedder:
    """gemini-embedding-001 batched async embedder.

    Gemini accepts up to 100 inputs per request; default to 96 for headroom.
    """

    DEFAULT_BATCH_SIZE = 96

    def __init__(
        self,
        *,
        model: str,
        output_dim: int,
        batch_size: int = DEFAULT_BATCH_SIZE,
        client: genai.Client | None = None,
    ) -> None:
        self._client = client or genai.Client()
        self._model = model
        self._batch_size = batch_size
        self._config = types.EmbedContentConfig(output_dimensionality=output_dim)

    async def embed(self, texts: Iterable[str]) -> list[list[float]]:
        return await _gather_batches(list(texts), self._batch_size, self._embed_batch)

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        resp = await self._client.aio.models.embed_content(
            model=self._model,
            contents=batch,
            config=self._config,
        )
        return [_l2_normalize(e.values or []) for e in (resp.embeddings or [])]


class OpenAIEmbedder:
    """text-embedding-3-* batched async embedder.

    OpenAI accepts up to 2048 inputs per request; default to 256 for memory
    headroom on long-document ingest.
    """

    DEFAULT_BATCH_SIZE = 256

    def __init__(
        self,
        *,
        model: str,
        output_dim: int,
        batch_size: int = DEFAULT_BATCH_SIZE,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI()
        self._model = model
        self._output_dim = output_dim
        self._batch_size = batch_size

    async def embed(self, texts: Iterable[str]) -> list[list[float]]:
        return await _gather_batches(list(texts), self._batch_size, self._embed_batch)

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(
            model=self._model,
            input=batch,
            dimensions=self._output_dim,
        )
        return [_l2_normalize(d.embedding) for d in resp.data]


_GEMINI_MODEL_PREFIXES = ("gemini-", "models/gemini")
_OPENAI_MODEL_PREFIXES = ("text-embedding-", "openai/text-embedding-")


def build_embedder(model_id: str, output_dim: int) -> Embedder:
    """Construct the embedder for an embedding-model identifier.

    Provider is inferred from the model id — ``gemini-*`` → Gemini,
    ``text-embedding-*`` (or ``openai/text-embedding-*``) → OpenAI.
    """
    if _is_gemini_model(model_id):
        return GeminiEmbedder(model=model_id, output_dim=output_dim)
    if _is_openai_model(model_id):
        return OpenAIEmbedder(
            model=model_id.removeprefix("openai/"),
            output_dim=output_dim,
        )
    raise ValueError(
        f"Unsupported embedding model: {model_id!r}. "
        "Expected a 'gemini-*' or 'text-embedding-*' identifier."
    )


def _is_gemini_model(model_id: str) -> bool:
    return model_id.lower().startswith(_GEMINI_MODEL_PREFIXES)


def _is_openai_model(model_id: str) -> bool:
    return model_id.lower().startswith(_OPENAI_MODEL_PREFIXES)


# ---------------------------------------------------------------------------


async def _gather_batches(
    texts: list[str],
    batch_size: int,
    embed_batch: Callable[[list[str]], Awaitable[list[list[float]]]],
) -> list[list[float]]:
    if not texts:
        return []
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    results = await asyncio.gather(*(embed_batch(b) for b in batches))
    return [v for r in results for v in r]


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]
