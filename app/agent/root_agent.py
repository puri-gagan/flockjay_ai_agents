"""Root LlmAgent factory.

Single agent Worflow, well-instructed, with:
 - Flockjay MCP tools
 - attachment tools

Switch llm provider and model based on available api key - tested with gemini-3-flash-preview
"""

from __future__ import annotations

import logging
import os
from typing import Union

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from app.agent.instructions import SYSTEM_INSTRUCTION
from app.agent.mcp_toolset import build_flockjay_mcp_toolset
from app.attachments.tools import build_attachment_tools
from app.constants import LLM_MODEL
from app.settings import settings

log = logging.getLogger(__name__)


def _resolve_model(model: str) -> Union[str, LiteLlm]:
    """Native string for Gemini, LiteLlm wrapper for everything else."""
    lowered = model.lower()
    if lowered.startswith("gemini-") or lowered.startswith("models/gemini"):
        return model
    return LiteLlm(model=model)


# Opik tracer — module-level, constructed once when OPIK_API_KEY is set.
tracer = None
if settings.opik_api_key:
    os.environ["OPIK_API_KEY"] = settings.opik_api_key
    os.environ["OPIK_WORKSPACE"] = settings.opik_workspace
    os.environ["OPIK_PROJECT_NAME"] = settings.opik_project_name
    try:
        from opik.integrations.adk import OpikTracer

        tracer = OpikTracer(project_name=settings.opik_project_name, tags=["flockjay"])
        log.info("Opik tracer attached (project=%s)", settings.opik_project_name)
    except Exception as exc:
        log.warning("Opik tracer init failed (%s); continuing without tracing.", exc)


def build_root_agent(session_note: str = "") -> LlmAgent:
    """Build a fresh root agent.

    Flockjay MCP auth is handled by ``mcp-remote`` (see
    :mod:`app.agent.mcp_toolset`); no per-request JWT is needed.
    """
    callbacks: dict = {}
    if tracer is not None:
        callbacks = {
            "before_agent_callback": tracer.before_agent_callback,
            "after_agent_callback": tracer.after_agent_callback,
            "before_model_callback": tracer.before_model_callback,
            "after_model_callback": tracer.after_model_callback,
            "before_tool_callback": tracer.before_tool_callback,
            "after_tool_callback": tracer.after_tool_callback,
        }

    instruction = (
        f"{SYSTEM_INSTRUCTION}\n\n{session_note}" if session_note else SYSTEM_INSTRUCTION
    )

    return LlmAgent(
        model=_resolve_model(LLM_MODEL),
        name="flockjay_rep_assistant",
        description=(
            "Sales enablement assistant for Flockjay reps — answers questions about "
            "deals, calls, content, coaching, teammates, and attachments."
        ),
        instruction=instruction,
        tools=[
            build_flockjay_mcp_toolset(),
            *build_attachment_tools(),
        ],
        **callbacks,
    )
