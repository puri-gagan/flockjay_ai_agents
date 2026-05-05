# Flockjay Agents

Agentic chat backend for Flockjay AI Agents. One `POST /chat` endpoint, one root agent, the Flockjay MCP server as the corpus, and an in-process attachment RAG.


|                    |                                                                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Framework**      | Google ADK 1.29.0 (`LlmAgent` + `Runner` + `MCPToolset`)                                                                                          |
| **API**            | FastAPI + Uvicorn, single worker, plain-text streaming response                                                                                   |
| **LLM**            | `LLM_MODEL` constant in `[app/agent/root_agent.py](app/agent/root_agent.py)`; Gemini routes natively, everything else via ADK's `LiteLlm` wrapper |
| **Attachment RAG** | Chroma in-memory + Gemini `gemini-embedding-001`, token-aware chunking via tiktoken                                                               |
| **MCP OAuth**      | `npx mcp-remote` stdio child (handles discovery, DCR, browser flow, token cache, silent refresh)                                                  |
| **Error recovery** | ADK `ReflectAndRetryToolPlugin` (subclassed) — converts MCP `isError:true`, raised tool exceptions, and hallucinated tool names into reflection guidance the model retries against |
| **Packaging**      | Poetry, Python 3.12                                                                                                                               |


---

## Table of contents

1. [Quick start](#quick-start)
2. [Configuration](#configuration)
3. [API reference](#api-reference)
4. [Architecture](#architecture)
5. [Operational notes & limitations](#operational-notes--limitations)
6. [Roadmap](#roadmap)
7. [MCP wishlist](#mcp-wishlist)
8. [Project structure](#project-structure)

---

## Quick start

```bash
# 1. Install
cd flockjay_agents
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
| `GEMINI_API_KEY`    | yes         | Always required — drives the Gemini attachment embedder; also the agent when `LLM_MODEL` is `gemini-*` (alias: `GOOGLE_API_KEY`) |
| `OPENAI_API_KEY`    | conditional | Required when `LLM_MODEL` starts with `openai/`                                                      |
| `ANTHROPIC_API_KEY` | conditional | Required when `LLM_MODEL` starts with `anthropic/`                                                   |
| `OPIK_API_KEY`      | optional    | Presence enables Opik tracing (see [Observability](#observability--opik))                            |
| `OPIK_WORKSPACE`    | optional    | Defaults to `default`                                                                                |
| `OPIK_PROJECT_NAME` | optional    | Defaults to `flockjay-agents`                                                                        |


### Switching LLM providers

The model is a hard-coded constant in `[app/agent/root_agent.py](app/agent/root_agent.py)`:

```python
LLM_MODEL: str = "gemini-2.5-flash"
# LLM_MODEL = "anthropic/claude-sonnet-4-5"   # needs ANTHROPIC_API_KEY
# LLM_MODEL = "openai/gpt-4o"                  # needs OPENAI_API_KEY
```

The resolver routes `gemini-*` through ADK's native path and everything else through ADK's `LiteLlm` wrapper. Note that `GEMINI_API_KEY` is always required regardless of `LLM_MODEL` because the attachment embedder is Gemini-only.

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
      {"id": "call-yesterday", "url": "http://localhost:9000/transcript.json", "type": "transcript"}
    ]
  }'
```

---

## Architecture

### One agent, LLM-driven routing

A single ADK `LlmAgent` holds the 18 Flockjay MCP tools plus 2 attachment tools and an instruction that tells the model how to choose. No programmatic router, no multi-agent mesh — for 20 tools with clear, non-overlapping intents that's unnecessary complexity.

Gemini 2.5 Flash supports parallel function calling natively, so multi-part questions ("compare X and tell me what the playbook says") issue multiple tool calls in a single turn.

### Per-request MCP toolset

Every `/chat` call rebuilds `MCPToolset(StreamableHTTPConnectionParams(...))` against the caller's auth. This scopes tenant auth to the request and guarantees we never hand one user's session to another's. The toolset is closed in a `finally` after each turn.

### Attachment handling — the interesting bit

Stuffing raw attachments into the LLM prompt would blow the context window and bury signal in noise. Two things happen instead:

1. **Pre-ingest at `/chat` time, outside the LLM loop.** Download → extract text (PDF via PyMuPDF, JSON transcripts by concatenating turns, subtitles by stripping cues) → token-aware chunk via tiktoken (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`) → batch embed via Gemini (`gemini-embedding-001`, L2-normalized) → register in an in-memory Chroma collection named `att_<sha256[:40]>`.
2. **Two agent tools wrap it:**
   - `search_attachment(attachment_id, query)` — top-k chunks for a specific query, with a similarity floor (`MIN_SIMILARITY=0.25`, `TOP_K=5`) so irrelevant attachments return `{ok: false, reason: "no chunks scored above similarity threshold"}` instead of noise.
   - `list_active_attachments()` — useful when the user refers to "the call" without naming an id.

The agent picks between targeted semantic search and enumeration; the instruction makes the tradeoff explicit. A short session note ("this session has attachments — use the attachment tools…") is appended to the agent's instruction whenever the registry has records for the session, so the model knows attachments exist without having to be told in the user message.

**Attachment as retrieval anchor.** The instruction tells the agent, when asked "find past calls where I handled this same objection", to first call `search_attachment` to extract the concrete objection phrasing, then feed that phrasing into `search_content` or `list_calls` against the Flockjay corpus. The attachment becomes the query; the MCP is the corpus.

**Conflict surfacing.** When chunks returned by `search_attachment` contradict retrieved playbook / content guidance from the MCP, the instruction tells the agent to present both views and flag the conflict explicitly rather than silently picking one.

### Sequential vs. parallel tool calls

Gemini's native parallel function calling handles this. We do not orchestrate it ourselves. The instruction tells the model to issue parallel tool calls when the question has independent parts, and to serialize only when one call's output feeds the next (e.g., `whoami` → `list_calls(author_id=me)`).

### Graceful degradation

Two distinct contracts feed the model:

- **Attachment tools** (`search_attachment`, `list_active_attachments`) return `{ok: true, …}` on success and `{ok: false, reason: "…"}` on empty / not-found / similarity threshold miss, with `available_attachment_ids` included when relevant.
- **Flockjay MCP tools** return raw entities (`whoami`) or DRF paginated dicts (`list_*`, `search_*`) on success. On a backend error, FastMCP server-side catches the exception and returns `CallToolResult(isError=true, content=[TextContent(text="Error executing tool …: <details>")])`. Empty result sets are `{count: 0, results: []}` — not an error.

The instruction bans fabrication explicitly: *"Do NOT invent deals, calls, scorecards, or content."* Backstop for the model "plowing ahead" on errors is the reflect-retry plugin below.

### Tool error recovery — reflect & retry plugin

ADK 1.29 ships an experimental `ReflectAndRetryToolPlugin` that intercepts tool failures, returns structured reflection guidance to the LLM (error details, args used, retry count, "consider these five things before your next attempt"), and lets the model self-correct. It tracks consecutive failures per-tool with an `asyncio.Lock` and a per-invocation counter. We register it on the `Runner` in [app/chat/router.py](app/chat/router.py).

The base plugin only sees **raised** exceptions through ADK's `on_tool_error_callback` path. That covers two of the three failure modes — hallucinated tool names (ADK's `_get_tool` raises `ValueError("Tool 'X' not found. Available tools: …")`), and transport / protocol-level failures from the MCP client. It misses the third: server-side tool errors come back as a *successful* `CallToolResult` with `isError: true`, because `MCPTool._run_async_impl` doesn't raise on those.

[app/agent/reflect_retry.py](app/agent/reflect_retry.py) closes that gap with a subclass that overrides `extract_error_from_result` to detect:

- `isError: true` on the MCP result dict — extracts the error text from `content[0].text`
- `ok: false` on the attachment-tool result dict — extracts the `reason`

Either match triggers the same reflection-and-retry path as a raised exception. `max_retries=2`, `throw_exception_if_retry_exceeded=False` so the stream never crashes — once the budget is exhausted, the plugin emits a "do not retry, try a different approach" message that the model is instructed to act on.

| Failure | Wire shape | Plugin path |
|---|---|---|
| LLM hallucinates a tool name | `ValueError("Tool 'X' not found …")` raised in ADK | base plugin's `on_tool_error_callback` |
| Flockjay backend 500 / 4xx | `{isError: true, content: [{text: "Error executing tool …"}]}` | overridden `extract_error_from_result` |
| MCP protocol-level error / transport dead | `McpError` / `BrokenPipeError` / etc. raised | base plugin's `on_tool_error_callback` |
| Attachment tool failure | `{ok: false, reason: "…"}` | overridden `extract_error_from_result` |
| Empty result set (`count: 0`) | `{count: 0, results: []}` | not retried — model handles per system prompt |

What this does **not** cover: re-establishing a dead `mcp-remote` session within the same turn. If the stdio child dies and ADK can't reconnect, every retry fails and the plugin emits the give-up message. Recovering that is a router-level concern (recreate the agent, recreate the toolset) — not yet wired.

### Stateful conversation

`InMemorySessionService` backs the ADK session. `runner.run_async(user_id, session_id, new_message)` replays prior events into each turn, so follow-ups like "what was that deal's stage history again?" answer from session memory without re-calling tools.

### Observability — Opik

Per-turn agent tracing via [Opik](https://www.comet.com/docs/opik/) is wired in but **off by default**. When enabled, every `/chat` request produces a single Opik trace containing all model calls, tool calls (MCP and attachment), and sub-agent spans, tagged with the `session_id` and `user_id`:

```python
from opik.integrations.adk import OpikTracer, track_adk_agent_recursive

tracer = OpikTracer(project_name="flockjay-agents", metadata={"session_id": ..., "user_id": ...})
track_adk_agent_recursive(agent, tracer)
# ... runner.run_async() ...
tracer.flush()
```

Setting `OPIK_API_KEY` is the toggle. The integration degrades gracefully: missing key, import error, or runtime failure logs a warning and falls back to no-op tracing — `/chat` is never blocked by observability.

---

## Operational notes & limitations

### Auth

- `/chat` is gated by a single shared `x-api-key` header (constant-time compared against `API_KEY`). For multi-tenant deployments, swap for a real per-user verifier (JWT + JWKS, etc.) and a per-user `MCPToolset` factory.
- Flockjay credentials are handled out-of-band by `mcp-remote`. The demo server exposes no JWKS, so JWT signature verification is currently limited to `exp`. Wire up `python-jose` with real public keys for production.

### Single-worker only

Session state, attachment registry, and artifact service are all in-memory. Horizontal scaling needs:

- An external session store (ADK ships `DatabaseSessionService`)
- A shared vector DB (Qdrant / pgvector)
- A shared artifact store

### Attachments

- **URLs only** — no file uploads. To test locally: `python -m http.server 9000` from a directory containing `transcript.json` / `document.pdf`, then pass `http://localhost:9000/transcript.json` as the attachment URL.
- **No STT for raw audio.** If the user attaches an audio file, we extract nothing useful. The Flockjay MCP returns pre-transcribed calls via `retrieve_calls`, so the workaround is *don't attach audio* — point the agent at the call id.
- **Synchronous ingestion.** A 50k-token transcript takes several seconds to embed on the first `/chat` call that references it. The user waits. Subsequent calls with the same `id` skip ingestion.
- **Session-scoped, no eviction.** Chunk embeddings live in process memory for the session's lifetime. Add a TTL / LRU when needed.

---

## Roadmap

In rough priority order:

1. **Persistent session store.** Swap `InMemorySessionService` for `DatabaseSessionService` (SQLite, zero infra) so the server can restart without losing conversations.
2. **Durable vector store.** Move attachments to Qdrant so "find similar calls from any previous session" becomes a tool.
3. **Streaming tool-call traces.** Body is already plain-text streamed; layered SSE event types would let clients render tool calls and partial reasoning, not just final text.
4. **Async attachment ingestion.** Separate `POST /attachments` endpoint that returns a job id; `/chat` references attachments by id and either waits or fails fast.
5. **Evals.** A `pytest` suite running the core agent scenarios against a dev MCP, asserting (a) the right tools were called and (b) no fabricated entities in the reply. Gate CI on it.
6. **Observability — extended.** Per-tool latency dashboards, span-level attachment metadata (chunk count, token count), evals-on-traces (LLM-as-judge), regression alerts.
7. **Rate limiting per JWT.** Token bucket keyed on `sub` claim — one rep can't exhaust Gemini/OpenAI quotas for the team.
8. **Cost accounting.** Sum prompt + completion tokens per session, surface in a response header.
9. **Audio STT.** Wire in Deepgram / Whisper for attached recordings the MCP doesn't already know about.

---

## MCP wishlist

Observed by inspecting actual tool responses against the underlying Flockjay REST API. The MCP wraps high-level shells but hides several endpoints the web app uses, which materially limits how well the agent can answer "what does this content actually say".

### Highest leverage — body / transcript access

- `**retrieve_document(document_id)` — the missing single tool.** The web app calls `GET /feed/document/{id}/` for the **full extracted text** of any uploaded media: call transcripts (with timestamped `segments[]`), PDF asset bodies, audio transcripts, OCR'd images. Document IDs are already in HTML returned by other tools as the `data-document-id` attribute on `<video>` / `<audio>` / `<embed>`. Wrapping that one endpoint unlocks quote-level coaching answers ("at 1:02 the rep said …") with zero external transcription.
- **External-embed dereference.** Lesson HTML often embeds Google Slides / Docs / Loom / YouTube / Vimeo via `<iframe>`. The MCP returns the iframe URL but no text. A `dereference_external_content(url)` — or a generic `web_fetch_text(url)` — would close the gap.
- `**retrieve_submodule(course_id, submodule_id)`.** `retrieve_learning_content(pk)` returns only the course shell with `content` set to the title. The web app uses `/api/course/{course_id}/` and `/api/course/{course_id}/submodule/{submodule_id}/`, which expose `submodule_type`, `assignment` data, `scorm_file`, `live_sessions[]`, `template_id`, and `user_progress.media_consumption` — none reachable today.
- **ID filter on `list_learning_content`.** Today the agent has to page through `list_learning_content` until `object_id` matches. A `learning_content_id` / `object_id` filter collapses that to one round-trip and composes cleanly with `expand=child_contents,content`.

### Quality of life

- **Server-side transcript search.** `search_within_call(call_id, query)` would skip a whole RAG detour for MCP-hosted calls.
- **Streaming tool responses.** `list_calls` with `page_size=100` is large; streaming would let the agent start synthesizing sooner.
- **Typed error codes.** Tool errors today are free-text; structured `{code, message}` would let wrappers branch reliably without regex.
- **Cursor-based pagination.** Page-number pagination forces N round-trips for "everything the rep has done this month".
- **Explicit "me" filter.** Nearly every query starts with `whoami`. A `me=true` short-circuit on filters would save the round trip.
- **Attachment as first-class content.** `ingest_attachment(url) -> content_id` would let us reuse Flockjay's existing indexing instead of running our own chunk store.
- **SCORM extraction.** `scorm_file` is opaque today; a manifest summary or extracted text would make compliance courses queryable.

---

## Project structure

```
flockjay_agents/
├── app/
│   ├── agent/
│   │   ├── instructions.py    # SYSTEM_INSTRUCTION
│   │   ├── mcp_toolset.py     # build_flockjay_mcp_toolset()  (stdio + mcp-remote)
│   │   ├── reflect_retry.py   # FlockjayReflectRetryPlugin (subclasses ADK's ReflectAndRetryToolPlugin)
│   │   └── root_agent.py      # build_root_agent() -> LlmAgent  (LLM_MODEL constant)
│   ├── attachments/
│   │   ├── embedder.py        # Gemini batched async embeddings (L2-normalized)
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
└── README.md
```