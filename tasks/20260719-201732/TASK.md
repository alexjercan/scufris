# Agent page: tools/model panel + per-turn tool-call & token display

- STATUS: CLOSED
- PRIORITY: 10
- TAGS: feature, backlog, agent, ui

## Goal

Enrich the agent (landing) page to show the agent's tools, model, and per-turn +
cumulative token/context usage alongside the conversation.

## Steps

- [x] `common.ts` types: extend `ChatReply` with `tool_calls: ToolCall[]` and
      `usage: TokenUsage | null`; add `ToolCall`, `TokenUsage`, `AgentInfo`,
      `AgentTool` interfaces (mirror the backend).
- [x] `index.html` (agent page): an agent bar in `chat__head` - model + a usage
      indicator + a "tools" toggle + the existing new-chat button - and a
      collapsible `#agent-tools` panel.
- [x] `agent-view.ts` (keep pure render helpers exported for tests):
      `renderAgentPanel(info, tools)` (model text, tools list + toggle),
      `messageMeta(reply)` (tool-call chips + the turn's token count, or null),
      `applyUsage(usage)` (accumulate output tokens + show last `input_tokens` as
      context) + `_resetAgentState()`. `startAgent` fetches `/api/agent/info` +
      `/api/agent/tools` and renders the panel when enabled; each reply appends
      the meta line and updates usage; new-chat resets. Escape tool/server names.
- [x] `style.css`: agent bar/meta, tools panel, message-meta chips - themed.
- [x] `agent-view.test.ts` (jsdom): `renderAgentPanel` (model + tools + hostile
      name escaped), `messageMeta` (chips + tokens), `applyUsage` (cumulative +
      context, reset).
- [x] LIVE serve smoke: the agent page shows the model + tools panel; a chat turn
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

## Implementation

- `common.ts`: `ChatReply` gains `tool_calls: ToolCall[]` + `usage: TokenUsage |
  null`; added `ToolCall`, `TokenUsage`, `AgentInfo`, `AgentTool` (mirror backend).
- `index.html`: an `agent-bar` in the chat head (model + usage indicator + a
  `tools` toggle + new-chat) and a collapsible `#agent-tools` panel.
- `agent-view.ts` (pure helpers exported for tests): `renderAgentPanel(info,
  tools)` (model text, tools list, toggle count/visibility), `messageMeta(reply)`
  (tool-call chips + the turn's token count, or null), `applyUsage(usage)`
  (cumulative output + last input_tokens as context) + `_resetAgentState()`.
  `startAgent` fetches `/api/agent/info` + `/api/agent/tools` and renders the
  panel when enabled; each reply appends the meta line + updates usage; new-chat
  resets. Tool/server names escaped. `style.css` themes the bar/panel/chips.
- Tests: 6 jsdom tests - renderAgentPanel (model + tools + hostile-name escape +
  hide-when-empty), messageMeta (chips + tokens, null when empty), applyUsage
  (cumulative + context + reset). 21 jsdom tests total; `npm run ci` green.

### Live verification (DoD)

Serve smoke on this host: the agent page carries the model/tools-toggle/tools-
panel/usage elements; `/api/agent/info` -> gpt-5.5, `/api/agent/tools` ->
[host_stats, tatr_ls, tatr_show]. Per-turn chips/tokens + cumulative indicator
are covered by the jsdom tests (the reply metadata was live-verified in tatr
20260719-201720).
