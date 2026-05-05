"""Plain-text attachment summarizer.

A small ADK ``LlmAgent`` running a lighter Gemini model. The full document text is sent in one call — no
chunking, no map-reduce, no JSON. The agent returns a few sentences of plain text.
"""

from __future__ import annotations

import logging
import uuid

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from app.constants import APP_NAME, SUMMARIZER_MODEL

log = logging.getLogger(__name__)

_INSTRUCTION = """You summarize long Documents mostly related to sales to short summarized version.
Capture the main themes, key stakeholders mentioned,
objections raised, and any decisions or outcomes. Do not invent content not
present in the input.
"""

SUMMARIZER_USER_ID = "summarizer"


class Summarizer:
    """One-shot plain-text summarizer backed by a lightweight ADK LlmAgent."""

    def __init__(self, model: str = SUMMARIZER_MODEL):
        self._agent = LlmAgent(
            name="attachment_summarizer",
            description="Summarizes an attachment into a few sentences of plain text.",
            model=model,
            instruction=_INSTRUCTION,
        )
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            app_name=APP_NAME,
            agent=self._agent,
            session_service=self._session_service,
        )

    async def summarize(self, text: str) -> str:
        if not text.strip():
            return ""

        session_id = str(uuid.uuid4())
        await self._session_service.create_session(
            app_name=APP_NAME, user_id=SUMMARIZER_USER_ID, session_id=session_id
        )
        try:
            new_message = Content(role="user", parts=[Part.from_text(text=text)])
            parts: list[str] = []
            async for event in self._runner.run_async(
                user_id=SUMMARIZER_USER_ID,
                session_id=session_id,
                new_message=new_message,
            ):
                if not event.is_final_response() or not event.content:
                    continue
                for p in event.content.parts or []:
                    if getattr(p, "text", None):
                        parts.append(p.text)
            return "".join(parts).strip()
        except Exception as exc:
            log.warning("Summarizer LlmAgent call failed: %s", exc)
            return ""
        finally:
            try:
                await self._session_service.delete_session(
                    app_name=APP_NAME, user_id=SUMMARIZER_USER_ID, session_id=session_id
                )
            except Exception:
                pass
