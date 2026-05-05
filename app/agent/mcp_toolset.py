"""Builder for the Flockjay MCP toolset.

Auth is delegated to ``mcp-remote`` (npm package) running as a stdio
child. mcp-remote handles OAuth — discovery, dynamic client registration,
PKCE, token cache, and silent refresh — and proxies the streamable-HTTP
MCP traffic over stdio. Tokens are cached on the host in ``~/.mcp-auth/``.

One-time bootstrap (per host)::

    npx -y mcp-remote https://api-demo.flockjay.com/mcp

Authorize in the browser. After that, this toolset is silent.
"""

from __future__ import annotations

from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp.client.stdio import StdioServerParameters

from app.constants import MCP_TIMEOUT_SECONDS
from app.settings import settings


def build_flockjay_mcp_toolset() -> MCPToolset:
    """ADK MCPToolset that talks to Flockjay via mcp-remote (stdio)."""
    return MCPToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx",
                args=["-y", "mcp-remote", settings.flockjay_mcp_url],
            ),
            timeout=MCP_TIMEOUT_SECONDS,
        ),
    )
