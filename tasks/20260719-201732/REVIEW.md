# Review: Agent page - tools/model panel + tool-call & token display

## Round 1 - 20260719

Scope: `web/src/common.ts` (types), `web/src/index.html` (agent bar + tools
panel), `web/src/agent-view.ts` (render helpers + wiring),
`web/src/agent-view.test.ts` (new), `web/src/style.css`.

### Correctness

- Delivers the spike's UI: a collapsible tools panel + the model in the chat
  head, per-assistant-message tool-call chips + the turn's token count, and a
  cumulative output-tokens / last-context indicator that resets on new chat.
- Serve-verified: the agent page carries the model/tools-toggle/tools-panel/usage
  elements and `/api/agent/info` + `/api/agent/tools` return the model and the
  three tools. The reply metadata itself was live-verified in the backend task
  (20260719-201720), so the end-to-end path (real reply -> chips) is covered by
  the sum of the two.
- The render logic is pure and jsdom-tested (6 new tests): `renderAgentPanel`
  (model text, tool list, toggle count, hide-when-empty, hostile-name escaped),
  `messageMeta` (chips + tokens, null when nothing), `applyUsage` (accumulates
  output, shows context, resets). The fetch-driven `startAgent`/`initChat` stay
  thin around those helpers - the side-effect-free-for-tests discipline holds.
- Escaping: tool/server names go through `escapeHtml` (a test proves a hostile
  tool name injects no element); chat message text still uses `textContent`.
- `ChatReply` gained required `tool_calls` + `usage` mirroring the backend; the
  chat still renders text-first (meta is additive), so a reply with no tools/usage
  simply shows no meta line. `npm run ci` (21 jsdom tests) green.

### Observations (non-blocking)

- MINOR: the cumulative indicator counts OUTPUT tokens across the session and shows
  the LAST turn's `input_tokens` as "context"; that is the spike's chosen signal
  (no per-model window from the CLI). A "% of window" bar would need a static
  per-model map - deferred as noted.
- MINOR: `messageMeta`/`renderAgentPanel` build escaped `innerHTML` (consistent with
  the rest of the codebase; the injection test covers it).
- NIT: reply metadata is per-turn (turn-based `codex exec`); live "tool
  running..." would need SSE, deferred in the spike.

### Verdict

- VERDICT: APPROVE

Meets the Definition of Done: the agent page shows the model + available
tools (collapsible), per-message tool chips + token count, and a cumulative
token/context indicator that resets on new chat; names escaped; jsdom tests +
`npm run ci` green; serve-verified. MINOR items are the spike's deferred polish.
