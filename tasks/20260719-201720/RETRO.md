# Retro: Surface agent tool-calls, token usage, model/tools

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- The spike's core insight - the data is already flowing and being dropped - made
  this a small, clean change: `_parse_thread_id` became `_parse_events` and
  harvested tool calls + usage from the SAME stdout the agent already read for the
  thread id. No new codex processes, no SSE.
- Probing the real event shapes in the spike (`turn.completed.usage` fields,
  `mcp_tool_call` item fields) meant the parser and its unit test were written
  against reality, and the live run matched exactly first try.
- Sourcing tool/model info from Scufris's own settings + `mcp.list_tools()`
  (rather than `codex mcp list`) was accurate and avoided the empty-without-`-c`
  trap the spike flagged.
- Default-empty new fields on `AgentReply`/`TurnOutcome` kept the chat panel,
  DisabledAgent, and all existing tests working with zero churn.

## What went wrong / friction

- Nothing notable. One small judgement: `/api/agent/tools` imports
  `scufris.mcp_server` (which builds a PsutilCollector at import) - harmless
  (cached import) and flagged in review as an optional tidy-up.

## Lessons

- `harvest-the-stream-you-already-run`: before adding endpoints or extra
  subprocess calls to expose a tool's internals, check what its existing output
  already carries - the agent's `codex exec --json` stream held tool calls +
  token usage that were being parsed for one field and discarded; extending the
  parse was nearly free.

## Follow-ups

- The frontend consumes this: agent-page tools/model panel + per-turn tool-call
  chips + cumulative token/context indicator (tatr 20260719-201732).
- Optional: a static per-model context-window map for a "% used" bar; SSE for
  live tool activity (both deferred in the spike).
