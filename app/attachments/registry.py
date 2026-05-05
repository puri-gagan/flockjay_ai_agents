"""In-memory attachment registry.

Holds, per session:
  - A Chroma collection of chunk embeddings for each attachment
  - Minimal metadata (content_type, source URL, chunk count)

The registry is process-local — a deliberate constraint of the
single-worker FastAPI deployment. To horizontally scale, replace the
Chroma `EphemeralClient` with a persistent vector store (Qdrant,
pgvector) and persist `_records` to a shared backing store.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

import chromadb
from chromadb.api.types import EmbeddingFunction

log = logging.getLogger(__name__)


@dataclass
class AttachmentRecord:
    attachment_id: str
    session_id: str
    source_url: str
    content_type: str
    chunk_count: int
    length_chars: int
    registered_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    collection_name: str = ""


class _NoopEmbeddingFunction(EmbeddingFunction):
    """Satisfies Chroma's API without downloading any model.

    We always supply `embeddings=` / `query_embeddings=` explicitly, so this
    never runs. If it ever does, it's a bug — raise loudly.
    """

    def __call__(self, input):
        raise RuntimeError(
            "Embeddings must be passed explicitly; Chroma should never "
            "invoke the default embedding function."
        )

    @staticmethod
    def name() -> str:  # Chroma 0.5+ requires this
        return "noop"


class AttachmentRegistry:
    def __init__(self) -> None:
        self._client = chromadb.EphemeralClient()
        # session_id -> attachment_id -> AttachmentRecord
        self._records: dict[str, dict[str, AttachmentRecord]] = {}
        # cache the embedding function instance so Chroma dedupes correctly
        self._ef = _NoopEmbeddingFunction()
        self._lock = threading.Lock()

    @staticmethod
    def _collection_name(session_id: str, attachment_id: str) -> str:
        digest = hashlib.sha256(f"{session_id}:{attachment_id}".encode()).hexdigest()[:40]
        return f"att_{digest}"

    def has(self, session_id: str, attachment_id: str) -> bool:
        return attachment_id in self._records.get(session_id, {})

    def register(
        self,
        *,
        session_id: str,
        attachment_id: str,
        source_url: str,
        content_type: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> AttachmentRecord:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        name = self._collection_name(session_id, attachment_id)
        with self._lock:
            # get_or_create so re-registration is idempotent
            coll = self._client.get_or_create_collection(name=name, embedding_function=self._ef)
            if chunks:
                coll.add(
                    ids=[f"{attachment_id}:{i}" for i in range(len(chunks))],
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=[
                        {"chunk_idx": i, "attachment_id": attachment_id}
                        for i in range(len(chunks))
                    ],
                )

            record = AttachmentRecord(
                attachment_id=attachment_id,
                session_id=session_id,
                source_url=source_url,
                content_type=content_type,
                chunk_count=len(chunks),
                length_chars=sum(len(c) for c in chunks),
                collection_name=name,
            )
            self._records.setdefault(session_id, {})[attachment_id] = record
        log.info(
            "Registered attachment session=%s id=%s chunks=%d",
            session_id, attachment_id, len(chunks),
        )
        return record

    def get(self, session_id: str, attachment_id: str) -> AttachmentRecord | None:
        return self._records.get(session_id, {}).get(attachment_id)

    def list_for_session(self, session_id: str) -> list[AttachmentRecord]:
        return list(self._records.get(session_id, {}).values())

    def drop_session(self, session_id: str) -> None:
        records = self._records.pop(session_id, {})
        for record in records.values():
            try:
                self._client.delete_collection(record.collection_name)
            except Exception as exc:  # pragma: no cover - chroma best-effort
                log.debug("delete_collection failed for %s: %s", record.collection_name, exc)

    def query(
        self,
        session_id: str,
        attachment_id: str,
        query_embedding: list[float],
        n_results: int,
    ) -> list[tuple[str, float, int]]:
        """Return [(chunk_text, similarity_score, chunk_idx), ...].

        `similarity_score` is in [0, 1] — we convert from Chroma's default
        L2 distance on normalized vectors via `1 - dist/2`. The Gemini
        embedder L2-normalizes vectors before storing
        """
        record = self.get(session_id, attachment_id)
        if record is None:
            return []

        coll = self._client.get_collection(record.collection_name, embedding_function=self._ef)
        res = coll.query(query_embeddings=[query_embedding], n_results=n_results)

        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        out: list[tuple[str, float, int]] = []
        for doc, dist, meta in zip(docs, dists, metas, strict=False):
            score = max(0.0, min(1.0, 1.0 - (dist or 0.0) / 2.0))
            idx = int((meta or {}).get("chunk_idx", -1))
            out.append((doc, score, idx))
        return out

    def build_session_note(self, session_id: str) -> str:
        """Short note appended to the system instruction when the session has attachments.

        The note exists so the model knows attachments are available without
        the user having to mention them in every turn. The full attachment
        catalog (ids, types) is fetched on demand via ``list_active_attachments``.
        """
        if not self.list_for_session(session_id):
            return ""
        return (
            "This session has one or more attachments. Use list_active_attachments "
            "to see them, then search_attachment to drill in based on the user's question."
        )


# Module-level singleton — wired up in app.runtime.runner
registry = AttachmentRegistry()
