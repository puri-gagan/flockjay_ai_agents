"""POST /chat — thin HTTP layer over :class:`app.chat.service.ChatService`.

The route resolves the request, delegates orchestration to the service, and
wraps the resulting async iterator in a ``StreamingResponse`` with the
session id surfaced in a response header.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.auth import require_api_key
from app.chat.schemas import ChatRequest
from app.chat.service import ChatService, get_chat_service

router = APIRouter()

ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


@router.post("/chat", dependencies=[Depends(require_api_key)])
async def chat(req: ChatRequest, service: ChatServiceDep) -> StreamingResponse:
    session_id, body = await service.start_stream(req)
    return StreamingResponse(
        body,
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Session-Id": session_id,
            "X-Accel-Buffering": "no",  # disables nginx response buffering
            "Cache-Control": "no-cache",
        },
    )
