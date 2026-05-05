"""Tool error → reflection / retry plugin for the Flockjay agent without crashing.

Wraps ADK's experimental ``ReflectAndRetryToolPlugin`` and teaches it the
two failure shapes the agent actually sees on the wire:

1. **MCP tool errors.** ``MCPTool._run_async_impl`` does not raise on
   tool-side failures

2. **Attachment tool failures.** ``app.attachments.tools`` returns
   ``{"ok": false, "reason": "..."}`` on failure — same problem.

LLM-hallucinated tool names and exceptions raised by either tool
flow through the base plugin's ``on_tool_error_callback`` path
(ADK's ``_get_tool`` raises ``ValueError`` with the available-tools list
embedded in the message).
"""

from __future__ import annotations

from typing import Any

from google.adk.plugins.reflect_retry_tool_plugin import (
    ReflectAndRetryToolPlugin,
    TrackingScope,
)
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext


def _extract_mcp_error_text(result: dict[str, Any]) -> str | None:
    """Return the error text from an MCP CallToolResult dump, or None."""
    if not result.get("isError"):
        return None
    parts = result.get("content") or []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if text:
                return str(text)
    return "MCP tool returned isError=true with no text content"


class FlockjayReflectRetryPlugin(ReflectAndRetryToolPlugin):
    async def extract_error_from_result(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(result, dict):
            return None

        mcp_error = _extract_mcp_error_text(result)
        if mcp_error is not None:
            return {"source": "mcp", "tool": tool.name, "error": mcp_error}

        if result.get("ok") is False:
            return {
                "source": "attachment_tool",
                "tool": tool.name,
                "error": result.get("reason", "ok=false with no reason"),
            }

        return None


def build_plugin(max_retries: int = 2) -> FlockjayReflectRetryPlugin:
    return FlockjayReflectRetryPlugin(
        name="flockjay_reflect_retry",
        max_retries=max_retries,
        throw_exception_if_retry_exceeded=False,
        tracking_scope=TrackingScope.INVOCATION,
    )
