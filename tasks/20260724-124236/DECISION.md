# Decision: carry the orchestrator session capabilities on the AgentBackend protocol (read_context sync, delete_session async)

- DATE: 20260724-124236
- STATUS: ACCEPTED
- TASK: 20260724-124236
- TAGS: decision, agents, sessions, backend

## Context

The orchestrator's session endpoints (`get_session_transcript`, `get_context`,
`fork_session`, `delete_agent_session`) read via the codex module functions
(`resolve_codex_home` + `read_transcript`/`read_context`/`delete_session`), so a
claude/opencode orchestrator could list sessions but not open, context, fork, or
truly delete them. The switcher list was already routed through the backend
(part 1). `read_transcript`/`read_status` already exist on the `AgentBackend`
protocol; context and delete did not.

## Decision

Add two capability methods to the `AgentBackend` protocol so `app.py` never
branches on backend:

- `read_context(settings, session_id) -> SessionContext | None` - **synchronous**,
  matching the other read methods. Codex delegates to the rich rollout reader
  (`sessions.read_context`, keeps cached/reasoning/total/window); claude and
  opencode map their `read_status` snapshot via `_context_from_status` (window 0
  - those backends expose no per-session window; honest, not lossy-by-bug); mock
  returns None.
- `delete_session(settings, session_id) -> bool` - **asynchronous**. opencode's
  delete is a daemon call that belongs on the async `OpencodeClient` (the WRITE
  path, alongside `create_session`/`send_message`), so this method is async to
  stay on that client boundary rather than opening a second blocking-httpx path
  or nesting an event loop. codex/claude do a local unlink in the async body (no
  await); mock returns False. The one caller, the app's `delete_agent_session`
  route, is already `async def` and awaits it.

## Alternatives considered

- **All-sync `delete_session` (blocking httpx in the backend, mirroring the read
  path's `_read_messages`)** - keeps the protocol uniform (all read/delete
  methods sync), but forks opencode into a second ad-hoc httpx path outside
  `OpencodeClient`, duplicating the auth/timeout/error handling the client
  centralises. Rejected: the client boundary is the cleaner seam for a write, and
  the sole call site is already async. This is the one asymmetry in the protocol
  (sync reads, async delete) and the reason for it.
- **Branch on backend in `app.py`** (codex module readers vs backend methods) -
  reintroduces exactly the codex-shaped coupling this task removes.
- **A `read_context` that always maps from `read_status`** - would drop codex's
  rich cached/reasoning/total breakdown; codex overrides to keep it.

## Consequences

- The four orchestrator session endpoints work on codex, claude, and opencode;
  `resolve_codex_home` remains only in the genuinely codex-specific
  usage/memory/account reads.
- The protocol is slightly non-uniform: reads are sync, delete is async.
  Accepted for the client-boundary reason above; every call site handles it
  (sync reads from sync routes, async delete from the async delete route).
- claude/opencode delete performs a real provider delete (file unlink / daemon
  `DELETE`); a backend with none returns False and the registry forget is the
  only cleanup (mock).
