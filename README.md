# Flockjay Agents

Agentic chat backend for Flockjay AI Agents. One `POST /chat` endpoint, one root agent, the Flockjay MCP server as the corpus, and an in-process attachment RAG.


|                    |                                                                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Framework**      | Google ADK 1.29.0 (`LlmAgent` + `Runner` + `MCPToolset`)                                                                                          |
| **API**            | FastAPI + Uvicorn, single worker, plain-text streaming response                                                                                   |
| **LLM**            | `LLM_MODEL` constant in `[app/agent/root_agent.py](app/agent/root_agent.py)`; Gemini routes natively, everything else via ADK's `LiteLlm` wrapper |
| **Attachment RAG** | Chroma in-memory + Gemini (`gemini-embedding-001`) or OpenAI (`text-embedding-3-*`), token-aware chunking via tiktoken                            |
| **MCP OAuth**      | `npx mcp-remote` stdio child (handles discovery, DCR, browser flow, token cache, silent refresh)                                                  |
| **Error recovery** | ADK `ReflectAndRetryToolPlugin` (subclassed) — converts MCP `isError:true`, raised tool exceptions, and hallucinated tool names into reflection guidance the model retries against |
| **Packaging**      | Poetry, Python 3.12                                                                                                                               |


---

## Table of contents

1. [Quick start](#quick-start)
2. [Configuration](#configuration)
3. [API reference](#api-reference)
4. [Project structure](#project-structure)

For architecture, known limitations, the improvement plan (including the persistent-session migration), the MCP wishlist, and the project layout, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Quick start

```bash
# 1. Install
cd flockjay_ai_agents
poetry install
cp .env.example .env
# Fill in:
#   API_KEY=$(openssl rand -hex 32)        # gates POST /chat
#   GEMINI_API_KEY=...                     # always required (Gemini embeddings; also the agent if LLM_MODEL is gemini-*)
#   OPENAI_API_KEY=... or ANTHROPIC_API_KEY=...   # only if LLM_MODEL points at that provider

# 2. Bootstrap the Flockjay MCP OAuth (one-time, per host)
npx -y mcp-remote https://api-demo.flockjay.com/mcp
# Authorize in the browser, wait for "Proxy established successfully", Ctrl+C.
# Tokens are cached at ~/.mcp-auth/ and silently refreshed thereafter.

# 3. Run
poetry run uvicorn app.main:app --reload --port 8000

# 4. Smoke check
curl http://localhost:8000/health
# {"status":"ok"}

# 5. Talk to the agent (streams plain text)
curl -N -X POST http://localhost:8000/chat \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "What does the playbook say about handling procurement objections?"}'
```

> **Production:** The browser-based `mcp-remote` bootstrap is for dev convenience, not a deploy strategy. For production, we need to do proper cache for a server-side OAuth client with proper JWT verification (signature against the issuer's JWKS etc).

---

## Configuration

### Environment


| Var                 | Required    | Purpose                                                                                              |
| ------------------- | ----------- | ---------------------------------------------------------------------------------------------------- |
| `API_KEY`           | yes         | Shared secret; constant-time compared against the `x-api-key` header on `/chat`                      |
| `GEMINI_API_KEY`    | conditional | Required when `LLM_MODEL` is `gemini-*` or `EMBEDDING_MODEL` is `gemini-*` (alias: `GOOGLE_API_KEY`)   |
| `OPENAI_API_KEY`    | conditional | Required when `LLM_MODEL` starts with `openai/` or `EMBEDDING_MODEL` is `text-embedding-*`            |
| `ANTHROPIC_API_KEY` | conditional | Required when `LLM_MODEL` starts with `anthropic/`                                                   |
| `OPIK_API_KEY`      | optional    | Presence enables Opik tracing (see [Observability](#observability--opik))                            |
| `OPIK_WORKSPACE`    | optional    | Defaults to `default`                                                                                |
| `OPIK_PROJECT_NAME` | optional    | Defaults to `flockjay-agents`                                                                        |


### Switching providers

Both the agent LLM and the attachment embedder are configured via constants in [app/constants.py](app/constants.py); the provider is inferred from the model string.

```python
# Agent LLM
LLM_MODEL: str = "gemini-3-flash-preview"
# LLM_MODEL = "anthropic/claude-sonnet-4-5"        # needs ANTHROPIC_API_KEY
# LLM_MODEL = "openai/gpt-4o"                       # needs OPENAI_API_KEY

# Attachment embedder
EMBEDDING_MODEL: str = "gemini-embedding-001"
# EMBEDDING_MODEL = "text-embedding-3-small"        # needs OPENAI_API_KEY
# EMBEDDING_MODEL = "text-embedding-3-large"        # needs OPENAI_API_KEY
```

[app/agent/root_agent.py](app/agent/root_agent.py) routes `gemini-*` through ADK's native path and everything else through ADK's `LiteLlm` wrapper. [app/attachments/embedder.py](app/attachments/embedder.py) routes `gemini-*` through `GeminiEmbedder` and `text-embedding-*` through `OpenAIEmbedder`. Both implementations satisfy the same `Embedder` protocol — the rest of the pipeline doesn't care which one is in use.

The output dimension (`EMBEDDING_OUTPUT_DIM = 768`) is honored by both providers and must be held constant for a given Chroma collection's lifetime.

---

## API reference

### `GET /health`

Open. Returns `{"status": "ok"}`.

### `POST /chat`

Gated by `x-api-key`. Streams the agent's reply as plain UTF-8 text (chunked via `StreamingResponse`, ADK `StreamingMode.SSE` internally).

**Request**

```json
{
  "session_id": "demo-1",
  "user_id": "optional-user-id",
  "message": "Find past calls where I handled this same pricing objection.",
  "attachments": [
    {"id": "call-yesterday", "url": "https://example.com/transcript.json", "type": "transcript"}
  ]
}
```

- `session_id` — optional. Auto-generated if omitted; either way, the resolved id is returned in the `X-Session-Id` response header. Reuse it across requests for stateful conversation.
- `attachments` — optional. Each is ingested on first reference and cached for the session; subsequent requests with the same `id` skip re-ingestion.
- `attachments[].url` — must be HTTP(S) reachable from the server. For local testing, drop a file into `./samples/` and reference it at `http://localhost:8000/samples/<filename>` (the app mounts that directory as static — no second web server needed).

**Response**

- `Content-Type: text/plain; charset=utf-8`
- `X-Session-Id: <session_id>`
- `X-Accel-Buffering: no`, `Cache-Control: no-cache` — for proxy compatibility.

The body is streamed text. Attachment ingestion failures are emitted inline as `[attachment error: ...]` before the model output. Use `curl -N` (or any client that doesn't buffer) to see tokens as they arrive.

**Example with attachment**

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-1",
    "message": "Find past calls where I handled this same pricing objection.",
    "attachments": [
      {"id": "call-yesterday", "url": "http://localhost:8000/samples/transcript.json", "type": "transcript"}
    ]
  }'
```

---

## Project structure

```
flockjay_ai_agents/
├── app/
│   ├── agent/
│   │   ├── instructions.py    # SYSTEM_INSTRUCTION
│   │   ├── mcp_toolset.py     # build_flockjay_mcp_toolset()  (stdio + mcp-remote)
│   │   ├── reflect_retry.py   # FlockjayReflectRetryPlugin (subclasses ADK's ReflectAndRetryToolPlugin)
│   │   └── root_agent.py      # build_root_agent() -> LlmAgent  (LLM_MODEL constant)
│   ├── attachments/
│   │   ├── embedder.py        # Embedder protocol + Gemini / OpenAI implementations + factory
│   │   ├── extractors.py      # PDF / transcript-JSON / subtitles / plain
│   │   ├── ingest.py          # download -> extract -> chunk -> embed -> register
│   │   ├── registry.py        # AttachmentRegistry + Chroma collection mgmt
│   │   └── tools.py           # FunctionTool wrappers (search_attachment, list_active_attachments)
│   ├── chat/
│   │   ├── router.py          # POST /chat (streams plain text)
│   │   └── schemas.py
│   ├── runtime/
│   │   └── runner.py          # lifespan singletons
│   ├── auth.py                # x-api-key dependency
│   ├── constants.py
│   ├── main.py                # FastAPI app + /health + /samples static mount
│   └── settings.py            # pydantic-settings
├── samples/                   # served at /samples/<file> for local attachment testing
├── pyproject.toml
├── .env.example
├── ARCHITECTURE.md            # this file
└── README.md
```

---

For architecture, known limitations & improvements, MCP wishlist, see [ARCHITECTURE.md](ARCHITECTURE.md).