# Agent page: tools/model panel + per-turn tool-call & token display

- STATUS: OPEN
- PRIORITY: 10
- TAGS: feature,backlog,agent,ui

## Goal

Enrich the agent (landing) page to show the agent's tools, model, and per-turn +
cumulative token/context usage alongside the conversation.

## Notes

- Spike: tasks/20260719-180528/SPIKE.md.
- Depends on the backend surfacing task (tatr 20260719-201720) for
  `/api/agent/info`, `/api/agent/tools`, and the `tool_calls`/`usage` on the chat
  reply.
- UI (web/src, agent page): a collapsible "agent" panel showing the model in use
  and the available tools/MCP (name + description; disabled/enabled state from
  /api/agent/info). Inline per assistant message: small chips for the tools that
  turn ran ("ran host_stats") and the turn's token count. A cumulative
  tokens / context indicator (accumulate per-turn usage client-side; show
  input_tokens as the context-fill signal).
- Keep host-derived strings escaped; keep the render module side-effect-free for
  jsdom tests; theme it like the rest.
- Open (from spike): panel layout/density (side panel vs header strip vs inline)
  is a /plan call; a static per-model context-window map for a "% used" bar is
  optional polish. SSE streaming of live tool activity is deferred.
