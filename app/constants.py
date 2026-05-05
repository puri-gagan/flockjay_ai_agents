"""Hard-coded application constants.

Anything that's a code-level decision — model identifiers, algorithm
parameters, size/time limits, internal identifiers — lives here.

Anything that legitimately varies per deployment — secrets, hosts, port,
log level, feature toggles — lives in :mod:`app.settings` and is loaded
from ``.env``.

If you're tempted to add an env override for one of these constants,
ask first whether the value really needs to differ between dev / staging
/ prod. Most don't.
"""

from __future__ import annotations

# Identity
APP_NAME: str = "flockjay"
USER_AGENT: str = "flockjay-agents/0.1.0"
DEFAULT_USER_ID: str = "default"  # TODO: in actual implementation with persistence session, change to the user id of user and should be taken from the token


# Models
# LLM to be used in root agent. Change this constant to switch providers
LLM_MODEL: str = "gemini-3-flash-preview"

# Gemini embedding model used by the attachment subsystem.
EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_OUTPUT_DIM: int = 768
EMBEDDING_BATCH_SIZE: int = 96  # gemini-embedding-001 accepts up to 100 inputs per request

SUMMARIZER_MODEL: str = "gemini-2.5-flash" # lighter and cheaper Gemini model for one-shot attachment summarization

# Attachment ingestion and registration
ATTACHMENT_MAX_BYTES: int = 50 * 1024 * 1024  # limiting to 50 MB hard cap per attachment
ATTACHMENT_DOWNLOAD_TIMEOUT_SECONDS: float = 60.0
ATTACHMENT_DOWNLOAD_CONNECT_TIMEOUT_SECONDS: float = 10.0

TIKTOKEN_ENCODING: str = "cl100k_base"  # gives good estimates for token counts
CHUNK_SIZE: int = 800  # tokens per chunk
CHUNK_OVERLAP: int = 100  # token overlap between adjacent chunks (somewhat prevents data loss accross chunks)

# Retrieval
TOP_K: int = 5  # chunks returned per attachment search
MIN_SIMILARITY: float = 0.25  # cosine similarity floor for hits

# MCP toolset (Flockjay)
MCP_TIMEOUT_SECONDS: float = 30.0
MCP_SSE_READ_TIMEOUT_SECONDS: float = 300.0
