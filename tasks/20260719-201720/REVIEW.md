# Review: Surface agent tool-calls, token usage, model/tools

## Round 1 - 20260719

Scope: `scufris/agent.py` (models + `_parse_events`), `scufris/app.py`
(`/api/agent/info`, `/api/agent/tools`), `tests/test_agent.py`,
`tests/test_app.py`.

### Correctness

- Proven live end to end: `/api/agent/info` returns the model/auth/enabled,
  `/api/agent/tools` returns the three real MCP tools, and a `/api/chat` turn
  using a tool carried `tool_calls: [{scufris/host_stats, completed}]` and real
  `usage` (input 47061, cached 34944, output 87, reasoning 32).
- The design is right: the agent already ran `codex exec --json` (for the thread
  id) and discarded the rest, so `_parse_events` just harvests what was already
  flowing - no extra codex processes, no SSE machinery. `mcp_tool_call` items and
  `turn.completed.usage` are exactly the probed shapes from the spike.
- `_parse_events` is robust: skips malformed lines, guards non-dict events and
  missing fields, and is unit-tested against a sample stream (tools + usage, a
  `not json` line in the middle). The fake-codex integration test proves the same
  through the real subprocess path.
- Tool/model info is sourced from Scufris (settings + `mcp.list_tools()`), not a
  redundant `codex mcp list` - matching the spike's recommendation and avoiding
  the empty-without-`-c` trap.
- Back-compat kept: `AgentReply`/`TurnOutcome`'s new fields default empty, so
  `DisabledAgent`, the chat panel, and existing tests are unaffected. ruff + mypy
  + pytest green.

### Observations (non-blocking)

- MINOR: `/api/agent/tools` imports `scufris.mcp_server` (which constructs a
  `PsutilCollector` at import) inside the handler; import is cached so it happens
  once, but it means a second collector instance exists process-wide. Harmless;
  could be a module-level import if preferred.
- MINOR: `TurnOutcome` (a NamedTuple) uses a shared mutable `[]` default for
  `tool_calls`; never mutated (the runner always supplies a fresh list), so it is
  safe, but a `None` default + normalize would be stricter.
- NIT: usage is per-turn; the cumulative/context indicator and the tools panel
  are the frontend task (20260719-201732). `input_tokens` is the context-fill
  signal, as scoped.

### Verdict

- VERDICT: APPROVE

Meets the Definition of Done: chat replies carry tool_calls + token
usage parsed from the stream the agent already runs, and `/api/agent/info` +
`/api/agent/tools` expose the model/auth/enabled and the tool registry;
live-verified with real data and covered by a parse unit test, a fake-codex
integration test, and endpoint tests. MINOR items are optional tidy-ups.
