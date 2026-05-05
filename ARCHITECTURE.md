# Flockjay Agents — Architecture & Internals

Engineering-facing documentation. For setup and the API surface, see [README.md](README.md).

## Table of contents

1. [Architecture](#architecture)
2. [Known limitations & improvements](#known-limitations--improvements)
3. [MCP wishlist](#mcp-wishlist)
4. [Project structure](#project-structure)

---

## Architecture

### One agent, LLM-driven routing

A single ADK `LlmAgent` workflow holding the Flockjay MCP tools and attachment tools, RAG for attachment, with retry reflect

Multi-agent / Plan and execute workflow would be overengineering — for 20 tools with clear, non-overlapping intents that's unnecessary complexity.

Models like gemini-3-flash-preview and openai models supports decision to do multiple - parallel function calling natively, so for multi-part questions - llm issues multiple tool call decision and we do call them in parallel using google adk.

### MCP toolset

[app/agent/mcp_toolset.py](app/agent/mcp_toolset.py) builds `MCPToolset(StdioConnectionParams(...))` over an `npx -y mcp-remote <FLOCKJAY_MCP_URL>` child. `mcp-remote` handles OAuth (discovery, dynamic client registration, PKCE, token cache, silent refresh) and proxies streamable-HTTP MCP traffic over stdio. Tokens are cached on the host at `~/.mcp-auth/`.

A fresh toolset is built per `/chat` call and closed by context manager `Runner.__aexit`__. The OAuth identity is the host's single cached token 

LIMITATION NOTE: For true multi-tenant deployment the bootstrap needs to be replaced with per-caller token resolution. For Production: We need to do proper cache for a server-side OAuth client with proper JWT verification (signature against the issuer's JWKS etc).

### Attachment handling

Stuffing raw attachments into the LLM prompt would blow the context window and bury signal in noise. Two things happen instead:

1. **Pre-ingest at `/chat` time, outside the LLM loop.** Download → extract text (PDF via PyMuPDF, JSON transcripts by concatenating turns, subtitles by stripping cues) → token-aware chunk via tiktoken (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`) → batch embed via the configured `Embedder` (Gemini or OpenAI) → register in an in-memory Chroma collection for VDB.
2. **Two agent tools wrap it:**
  - `search_attachment(attachment_id, query)` — top-k chunks for a specific query, with a similarity floor (`MIN_SIMILARITY=0.25`, `TOP_K=5`) so irrelevant attachments return `{ok: false, reason: "no chunks scored above similarity threshold"}` instead of noise.
  - `list_active_attachments()` — useful when the user refers something from attachment sent in previous chat.

The agent picks between targeted semantic search and enumeration. 

A short session note ("this session has attachments — use the attachment tools…") is appended to the agent's instruction whenever the registry has records for the session, so the model knows attachments exist without having to be told in the user message and does things / responds based on the user intent

**Conflict surfacing.** When chunks returned by `search_attachment` contradict retrieved playbook / content guidance from the MCP, the instruction tells the agent to present both views and flag the conflict explicitly rather than silently picking one.

LIMITATION NOTE: The Chroma `EphemeralClient` is an in-process, in-memory vector store — chunks and embeddings are wiped on restart, can't be shared across uvicorn workers, and grow unbounded for the session's lifetime.

For production we should move to a hosted VDB (Qdrant, pgvector, Pinecone, etc.) so embeddings are durable, multi-worker safe, and reusable across sessions.

### Tool error recovery — reflect & retry

ADK has an `ReflectAndRetryToolPlugin` that intercepts tool failures, returns structured reflection guidance to the LLM (error details, args used, retry count, "consider these five things before your next attempt"), and lets the model self-correct. We've it register on the `Runner` in [app/chat/router.py](app/chat/router.py).

The base plugin only sees **raised** exceptions through ADK's `on_tool_error_callback` path. That covers two of the three failure modes — hallucinated tool names (ADK's `_get_tool` raises `ValueError("Tool 'X' not found. Available tools: …")`), and transport / protocol-level failures from the MCP client. 

But it misses the third: server-side tool errors come back as a *successful* `CallToolResult` with `isError: true`, because `MCPTool._run_async_impl` doesn't raise on those.

[app/agent/reflect_retry.py](app/agent/reflect_retry.py) FlockjayReflectRetryPlugin - closes that gap with a subclass that overrides `extract_error_from_result` to detect:

- `isError: true` on the MCP result dict — extracts the error text from `content[0].text`
- `ok: false` on the attachment-tool result dict — extracts the `reason`

LIMITATION NOTE: What this does **not** cover - re-establishing a dead `mcp-remote` session within the same turn.

### Stateful conversation

`InMemorySessionService` backs the ADK session. Sessions are keyed by `(app_name, user_id, session_id)`  replays prior events into each turn, so follow-ups like "what was that deal's stage history again?" answer from session memory without re-calling tools.

LIMITATION NOTE: `InMemorySessionService` is used here for demonstration purposes only — sessions live in process memory, are wiped on restart, aren't shared across uvicorn workers, and grow unbounded per session. 

For production we should swap in `DatabaseSessionService` so sessions are durable, multi-worker safe, and resumable across deploys.

### Context window management — event summarization

A long conversation replays the entire event history into every turn. Without compaction, the prompt grows linearly with turn count until it blows the model's context window or runs up the token bill. At the end of every invocation the `Runner` checks the configured triggers and, when either fires, asks a summarizer to fold older events into a single compacted event. Subsequent turns replay the summary instead of the raw history.

We configure both triggers ([app/constants.py](app/constants.py)):


| Trigger                       | Tunable                                                          | Default    | Behavior                                                                                                                                         |
| ----------------------------- | ---------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Token threshold** (primary) | `COMPACTION_TOKEN_THRESHOLD` / `COMPACTION_EVENT_RETENTION_SIZE` | 80 000 / 4 | When the last prompt's `prompt_token_count` crosses the threshold, summarize everything older than the last N raw events.                        |
| **Sliding window** (backstop) | `COMPACTION_INTERVAL` / `COMPACTION_OVERLAP_SIZE`                | 50 / 2     | Every N user invocations, summarize older events keeping the last K raw for recency. Runs only when token-threshold did **not** fire that cycle. |


Summaries are produced by `gemini-2.5-flash` independently from `LLM_MODEL`, so summarization stays cheap even when the agent itself runs a Pro/more expensive model. Compacted events are persisted on the session as `EventActions(compaction=EventCompaction(...))` and replayed in subsequent turns instead of the raw history.

### Observability — Opik

Per-turn agent tracing via [Opik](https://www.comet.com/docs/opik/) is wired in. Every `/chat` request produces a single Opik trace containing all model calls, tool calls (MCP and attachment), input and output with token usages, estimate pricing, KV cache cost etc.

The integration is wired in [app/agent/root_agent.py](app/agent/root_agent.py) — when `OPIK_API_KEY` is set, the module instantiates an `OpikTracer` once and passes its `before`** / `after`** hooks (agent, model, tool) as constructor callbacks on the root `LlmAgent`. Setting `OPIK_API_KEY` is the only toggle. The integration degrades gracefully: missing key, import error, or runtime failure logs a warning and falls back to no-op tracing — `/chat` is never blocked by observability.

---

## Known limitations & improvements

Each entry pairs a current constraint with the migration plan to lift it.

### Persistent session store (highest priority)

**Problem.** Sessions live in `InMemorySessionService` ([app/runtime/runner.py](app/runtime/runner.py)) — wiped on every restart, and not shared across uvicorn workers (a follow-up routed to a different worker hits an empty session). Single biggest scalability blocker.

**Fix.** Swap for ADK `DatabaseSessionService(db_url=...)` — SQLAlchemy async, Postgres (`postgresql+asyncpg://`) for prod. After the swap, any worker can resume any session by id.

### Process-local attachment registry

**Current.** Chunks + embeddings live in an in-process Chroma `EphemeralClient` keyed by `(session_id, attachment_id)` ([app/attachments/registry.py](app/attachments/registry.py)). 

Lost on restart; can't be served from a different worker than the one that ingested. Session-scoped with no eviction — embeddings live in process memory for the session's lifetime.

**Improvement plan.** Move to Qdrant, Pinecone or pgvector (the same Postgres if we adopt it for sessions).

### Auth

- `/chat` is gated by a single shared `x-api-key` header (constant-time compared against `API_KEY`). For multi-tenant deployments, swap for a real per-user verifier (JWT + JWKS) and a per-user `MCPToolset` factory.
- Flockjay credentials are handled out-of-band by `mcp-remote`.

### Attachments — operational

- **Synchronous ingestion.** A 50k-token transcript takes several seconds to embed on the first `/chat` call that references it. The user waits. Subsequent calls with the same `id` skip ingestion.

### Other improvements

- **Async attachment ingestion.** Separate `POST /attachments` endpoint that returns a job id; `/chat` references attachments by id and either waits or fails fast.
- **Evals.** YAML scenario suite (`evals/scenarios/*.yaml`) — each scenario pins a `user_message`, mocked MCP / attachment tool responses, and expected assertions: which tools the agent calls and with what args, plus an LLM-as-judge prompt scoring the final reply against criteria. **What this catches:** regressions from prompt edits, tool description / arg-schema changes, model upgrades, or routing logic tweaks — e.g. agent stops calling `search_content` for "what's our pricing" queries, drops the conflict-surfacing instruction, or starts hallucinating deal stages after a system-instruction rewrite. Without evals these only surface as user complaints in production; with evals they fail CI before merge.

---

## MCP Improvements

Observed by inspecting actual tool responses against the underlying Flockjay REST API. The MCP wraps high-level shells but hides several information, mostly inside deep filters like pagination, child expand etc., which materially limits how well the agent can answer "what does this content actually say".

**Responses today look like raw web-app API payloads, not agent-shaped context.** Like that React app consumes, with significant noise the LLM has to wade through:

**What we want instead:** simpler args (paginations, expand etc can be imporved?) and to-the-point response shapes (markdown text bodies, content from documents, call transcripts, group/tag IDs). Smaller payloads = lower token cost per turn, fewer distractor fields for the model to misread.

For example:

- **ID filter on `list_learning_content`.** Today the agent has to page through `list_learning_content` until `object_id` matches. A `learning_content_id` / `object_id` filter collapses that to one round-trip. Better still: when the user's intent is clearly "give me this item's content" (single-id lookup, retrieve_*), the response should return the body text by default — no `expand=child_contents,content` dance. The agent shouldn't have to know the right knob to turn; the tool should infer "you asked for it by id, you want the content" and serve it.

What the agent actually needs (I would expect it would be possible):

```
retrieve_learning_content(id="umsnb5u1ul5u7c")
→ {
    id, title, summary,                          // one-paragraph summary, plain text
    body_md,                                     // course body as markdown
    submodules: [
      { id, title, type, body_md, transcript_md } // each lesson's actual teaching content
    ],
    tags, author: {name, role}                   // minimal author/identity for citation
  }
```

No HTML, no `group_ids`, no `sf_metadata`, no `custom_fields`, no `expand` knob. One tool call, intent-inferred response, body text in markdown ready to feed the LLM.