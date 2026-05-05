"""Pydantic request models for POST /chat.

The endpoint streams plain text — there is no response model. Tool calls,
attachment statuses, and final-reply text are not buffered into JSON.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Attachment(BaseModel):
    """A file the user attaches to the conversation.

    We do NOT accept raw bytes — the caller passes a URL the server can GET.
    """

    id: str = Field(..., description="Stable caller-supplied id; used for dedup and tool refs.")
    url: str = Field(..., description="HTTP(S) URL the server can fetch.")
    type: Literal["pdf", "transcript", "text", "call_recording"] | None = Field(
        default=None,
        description="Optional hint. If omitted, the server infers from Content-Type and extension.",
    )


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The rep's question or command.")
    session_id: str | None = Field(
        default=None,
        description="Reuse an existing session for stateful conversation; server generates one if omitted.",
    )
    user_id: str | None = Field(
        default=None,
        description="Optional. Defaults to 'default' — ADK uses this for session scoping.",
    )
    attachments: list[Attachment] | None = Field(default=None)
