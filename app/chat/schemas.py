"""Pydantic request models for POST /chat.

The endpoint streams plain text — there is no response model. Tool calls,
attachment statuses, and final-reply text are not buffered into JSON.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Attachment(BaseModel):
    """A file the user attaches to the conversation.

    We do NOT accept raw bytes — the user passes a URL the server can GET.
    """

    id: str = Field(..., description="Stable caller-supplied id; used for dedup and tool refs.")
    url: str = Field(..., description="HTTP(S) URL the server can fetch.")
    type: Optional[Literal["pdf", "transcript", "text", "call_recording"]] = Field(
        default=None,
        description="Optional hint. If omitted, the server infers from Content-Type and extension.",
    )


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The rep's question or command.")
    session_id: Optional[str] = Field(
        default=None,
        description="Reuse an existing session for stateful conversation; server generates one if omitted.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional. Defaults to 'default' — ADK uses this for session scoping.",
    )
    attachments: Optional[list[Attachment]] = Field(default=None)
