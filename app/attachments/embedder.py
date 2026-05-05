"""Batched Gemini embedding helper.

Uses the ``google-genai`` SDK. Vectors are L2-normalized before return so
the registry's ``1 - dist/2`` cosine-similarity proxy on Chroma's L2
distance stays correct — Gemini does not pre-normalize outputs below
3072 dimensions.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Iterable

from google import genai
from google.genai import types

from app.constants import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, EMBEDDING_OUTPUT_DIM

log = logging.getLogger(__name__)


class GeminiEmbedder:
    def __init__(self, client: genai.Client, model: str = EMBEDDING_MODEL):
        self._client = client
        self._model = model
        self._config = types.EmbedContentConfig(output_dimensionality=EMBEDDING_OUTPUT_DIM)

    async def embed(self, texts: Iterable[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []

        batches: list[list[str]] = [
            texts[i : i + EMBEDDING_BATCH_SIZE]
            for i in range(0, len(texts), EMBEDDING_BATCH_SIZE)
        ]
        results = await asyncio.gather(*(self._embed_batch(b) for b in batches))
        out: list[list[float]] = []
        for r in results:
            out.extend(r)
        return out

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        resp = await self._client.aio.models.embed_content(
            model=self._model,
            contents=batch,
            config=self._config,
        )
        embeddings = resp.embeddings or []
        return [_l2_normalize(e.values or []) for e in embeddings]


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]
