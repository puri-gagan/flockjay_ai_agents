"""Hard-coded application constants.
"""

from __future__ import annotations

# Identity
APP_NAME: str = "flockjay"
USER_AGENT: str = "flockjay-agents/0.1.0"
# Fallback when ChatRequest.user_id is omitted. Once a real auth layer
# (JWT/JWKS) lands, derive the user id from the verified token instead.
DEFAULT_USER_ID: str = "default"


# Models
# Root-agent LLM. Change to switch providers — `gemini-*` routes through
# ADK's native path, anything else through ADK's LiteLlm wrapper.
LLM_MODEL: str = "gemini-3-flash-preview"

# Gemini embedding model used by the attachment subsystem.
EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_OUTPUT_DIM: int = 768
EMBEDDING_BATCH_SIZE: int = 96  # gemini-embedding-001 accepts up to 100 inputs per request

# Attachment ingestion
ATTACHMENT_MAX_BYTES: int = 50 * 1024 * 1024  # 50 MB hard cap per attachment
ATTACHMENT_DOWNLOAD_TIMEOUT_SECONDS: float = 60.0
ATTACHMENT_DOWNLOAD_CONNECT_TIMEOUT_SECONDS: float = 10.0

TIKTOKEN_ENCODING: str = "cl100k_base"  # gives good estimates for token counts
CHUNK_SIZE: int = 800  # tokens per chunk
CHUNK_OVERLAP: int = 100  # token overlap between adjacent chunks; reduces boundary-loss on retrieval

# Attachment retrieval
TOP_K: int = 5  # chunks returned per attachment search
MIN_SIMILARITY: float = 0.25  # cosine similarity floor for hits

# Flockjay MCP toolset
MCP_TIMEOUT_SECONDS: float = 30.0
MCP_SSE_READ_TIMEOUT_SECONDS: float = 300.0
