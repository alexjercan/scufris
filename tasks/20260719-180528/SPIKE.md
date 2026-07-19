# Spike: agent chat page - surface tools, MCPs, usage, context

- DATE: 20260719-180528
- STATUS: RECOMMENDED
- TAGS: spike, backlog, agent, ui

## Question

How do we enrich the agent chat page to surface the agent's internals - available
tools, MCP servers, token/context usage, the active model, and any other detail
the `codex` CLI can provide - usefully alongside the conversation? Which data is
actually available, how do we get each, and how do we show it?

## Context

The agent is `CodexCliAgent` driving `codex exec` (nixpkgs codex) with the Scufris
MCP server (`host_stats`, `tatr_ls`, `tatr_show`) registered per-invocation via
`-c` (task 20260719-162419). The chat panel (task 20260719-162406) shows only the
message turns; `AgentReply` currently carries just `{text, status}`. Crucially the
agent ALREADY runs `codex exec --json` (to recover the `thread_id` for continuity)
and reads the reply from `--output-last-message` - so it already has the event
stream in hand and simply discards the rest.

## What's actually available (probed, 2026-07-19)

`codex exec --json` (with `stdin=/dev/null`) emits, per turn:

- `thread.started` -> `{thread_id}` (already parsed).
- `turn.started`.
- `item.started` / `item.completed` with `item.type`:
  - `agent_message` -> the reply text.
  - `mcp_tool_call` -> `{server, tool, arguments, result, status, error}` (e.g.
    server=`scufris`, tool=`host_stats`, status=`completed`). This is the live
    tool activity.
  - `reasoning` (present, not needed for v1).
- `turn.completed` -> `{usage: {input_tokens, cached_input_tokens, output_tokens,
  reasoning_output_tokens}}`. Real per-turn token usage. (`input_tokens` is the
  context sent this turn - the practical "how full is context" signal.)

Other surfaces:

- **Model**: a Scufris setting (`agent_model`), NOT queried from codex - Scufris
  already knows it.
- **MCP tools**: Scufris DEFINES them (`scufris/mcp_server.py`) - it is the source
  of truth (name + description). `codex mcp list` is EMPTY without the
  per-invocation `-c` overrides (we register per call, nothing persisted in
  `~/.codex`), so it is not a reliable enumeration source.
- **Context window** (max tokens per model): not cleanly exposed by the codex CLI.
  Use `input_tokens` (context actually sent) as the fill signal; a static
  per-model window map is optional polish.

## Options considered

### Where the data comes from

- **Parse the `--json` stream the agent already runs (RECOMMENDED).** Extend the
  existing parse in `_run_codex_exec` to also collect `mcp_tool_call` items and
  the `turn.completed` usage, and return them on `AgentReply`. Zero extra codex
  processes; per-turn tool activity + token usage for free. Cons: it is a per-turn
  SUMMARY (all a turn's events arrive together), not a live stream.
- **Extra `codex` calls (rejected as the primary).** `codex mcp list -c <our
  overrides>` re-derives tools Scufris already owns; there is no CLI "list models"
  for a context window. Redundant and slower.
- **SSE stream events to the client (deferred).** Nice for live "tool running..."
  feedback, but `codex exec` is turn-based (the tool events land within the one
  turn), so a per-turn summary is enough for v1; SSE is a later enhancement if the
  turns feel slow.

### Where tool/model info comes from

- **Scufris's own registry + settings (RECOMMENDED)** - list the MCP tools from
  `mcp_server.py` (name + description) and the model/auth from settings. Accurate,
  no codex round-trip. vs `codex mcp list` (needs the `-c` overrides, redundant).

### Context fill

- **`input_tokens` per turn (RECOMMENDED)** - real data, "context sent this turn".
  vs a static per-model max-token map for a % bar (optional polish) vs nothing.

## Recommendation

1. **Backend - surface per-turn metadata from the stream we already parse.**
   Extend `_run_codex_exec` to collect `mcp_tool_call` items (`server`, `tool`,
   `status`) and the `turn.completed` `usage`, and add them to `AgentReply`
   (`tool_calls: list`, `usage: {...}`). `POST /api/chat` returns them with each
   reply. Add `GET /api/agent/info` (model, auth mode, enabled) and
   `GET /api/agent/tools` (the Scufris MCP tools: name + description), both sourced
   from Scufris, not codex.

2. **Frontend - enrich the agent (landing) page.** A collapsible "agent" panel
   showing the model in use and the available tools/MCP (from `/api/agent/tools`,
   with a disabled/enabled state). Inline per-message: small chips for the tools a
   turn ran ("ran host_stats") and the turn's token count. A cumulative
   tokens / context indicator (client accumulates per-turn `usage`; show
   `input_tokens` as the context-fill signal).

This beats the runners-up because the richest data (tool calls + token usage) is
already flowing through the agent and just being dropped - capturing it is nearly
free and needs no extra codex processes; tool/model info comes from Scufris's own
source of truth rather than a redundant `codex mcp list`. It fits the existing
turn-based chat (per-turn summary), leaving SSE and a context-window bar as clearly
scoped later polish.

## Open questions

- UI density / layout on the landing page: a persistent side panel vs a
  collapsible header strip vs inline-only chips - a `/plan`-time call; keep it
  from cluttering the chat.
- Whether to add a static per-model context-window map for a "% of context used"
  bar, or just show raw `input_tokens`. Start with raw; add the map if wanted.
- `reasoning` items and streaming (SSE) are deferred; revisit if turns feel slow
  or opaque.

## Next steps

Direction-level tasks this spike seeded, for `/plan` to break into steps:

- tatr 20260719-201720: backend - capture `mcp_tool_call` + `turn.completed`
  usage from the `--json` stream onto `AgentReply` / `POST /api/chat`, and add
  `GET /api/agent/info` + `GET /api/agent/tools` (sourced from Scufris).
- tatr 20260719-201732: frontend - agent page panel (model + available tools) +
  per-turn tool-call chips and token count + a cumulative token/context indicator.

## Fix record

(Appended by each implementing task as it lands.)
