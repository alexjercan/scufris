# Spike: agent chat page - surface tools, MCPs, usage, context

- PRIORITY: 0
- TAGS: spike, backlog, agent, ui
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Question

How do we enrich the agent chat page to surface the agent's internals - the
available tools, the configured MCP servers and their status, token/context
usage, the active model, and any other detail the `codex` CLI can provide - in a
useful, intuitive way alongside the conversation?

## Context

Today the chat page shows only the message turns. Under the hood the agent is
`CodexCliAgent` driving `codex exec` (nixpkgs codex), with the Scufris MCP server
(`host_stats`, `tatr_ls`, `tatr_show`) registered per-invocation via `-c`.
`codex exec --json` already emits structured events - `thread.started`,
`turn.started/completed`, `item.started/completed` (including `mcp_tool_call`
with server/tool/args/result), and token-usage figures ("tokens used"). Other
codex surfaces: `codex mcp list` (configured servers/tools), model info, session
id. So a lot is available; the spike is what to expose and how.

## What a good answer looks like

An inventory of what the codex CLI can provide (parse `codex exec --json` events
for tool calls + token usage; `codex mcp list` for tools/servers; model/context
window; session/thread id) and HOW to get each, plus a UI direction for showing
it: a tools/MCP panel (what tools exist, which server, enabled/health), live
tool-call activity inline in the chat (which tool ran, args, result), per-turn
and cumulative token/context usage, the model in use. Concrete enough to seed
implementation tasks.

## Candidate directions to explore (diverge before converging)

- Data source: parse the `--json` event stream Scufris already runs (the agent
  currently discards non-final events) vs extra `codex` calls (`codex mcp list`,
  model list). Stream events to the client (SSE) vs return a per-turn summary.
- Backend shape: extend the chat endpoint to return tool-call + usage metadata
  with each reply, and add a `/api/agent/tools` (from `codex mcp list` + our MCP
  server) and `/api/agent/info` (model, context window).
- UI: a side panel vs inline chips; how much to show without clutter; how it
  fits the multi-page restructure (the agent is the landing page).

## Notes

- Output per the /spike skill: write `tasks/<id>/SPIKE.md`, seed implementation
  tasks, close the spike.
- Builds on the agent backend (tatr 20260719-162356), chat panel
  (tatr 20260719-162406) and MCP server (tatr 20260719-162419). The agent already
  parses `thread.started` for continuity, so extending the event parse is natural.
- User ask (2026-07-19): show available tools, MCPs, usage, context - any detail
  codex can provide.
