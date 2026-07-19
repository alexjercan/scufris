# Agent page: tools/model panel + per-turn tool-call & token display

- STATUS: OPEN
- PRIORITY: 10
- TAGS: feature,backlog,agent,ui

## Goal

Enrich the agent (landing) page to show the agent's tools, model, and per-turn +
cumulative token/context usage alongside the conversation.

## Steps

- [ ] `common.ts` types: extend `ChatReply` with `tool_calls: ToolCall[]` and
      `usage: TokenUsage | null`; add `ToolCall`, `TokenUsage`, `AgentInfo`,
      `AgentTool` interfaces (mirror the backend).
- [ ] `index.html` (agent page): an agent bar in `chat__head` - model + a usage
      indicator + a "tools" toggle + the existing new-chat button - and a
      collapsible `#agent-tools` panel.
- [ ] `agent-view.ts` (keep pure render helpers exported for tests):
      `renderAgentPanel(info, tools)` (model text, tools list + toggle),
      `messageMeta(reply)` (tool-call chips + the turn's token count, or null),
      `applyUsage(usage)` (accumulate output tokens + show last `input_tokens` as
      context) + `_resetAgentState()`. `startAgent` fetches `/api/agent/info` +
      `/api/agent/tools` and renders the panel when enabled; each reply appends
      the meta line and updates usage; new-chat resets. Escape tool/server names.
- [ ] `style.css`: agent bar/meta, tools panel, message-meta chips - themed.
- [ ] `agent-view.test.ts` (jsdom): `renderAgentPanel` (model + tools + hostile
      name escaped), `messageMeta` (chips + tokens), `applyUsage` (cumulative +
      context, reset).
- [ ] LIVE serve smoke: the agent page shows the model + tools panel; a chat turn
      shows tool chips + a token count and updates the cumulative indicator.
      `ruff`/`mypy`/`pytest` + `npm run ci` green.

## Definition of Done

- The agent page shows the model in use and the available tools (collapsible
  panel), per assistant message the tools that turn ran + its token count, and a
  cumulative token / context indicator that updates each turn and resets on new
  chat. Names escaped; jsdom tests + `npm run ci` + python green; serve-verified.

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
