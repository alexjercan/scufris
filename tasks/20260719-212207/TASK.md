# Agent page: context breakdown + weekly-usage panel

- PRIORITY: 20
- TAGS: feature, agent, ui, spike
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Two read-only panels on the agent page, fed by the backend:

1. A `/context`-style view for the current session: context-window size, used %
   (a bar), cached-vs-fresh input tokens, cumulative output/reasoning, turn count
   and per-tool call counts.
2. A weekly-usage meter (in the sidebar): `rate_limits.primary.used_percent` over
   the 10080-minute (weekly) window, `resets_at` ("resets in 2d 5h"), plan type,
   and the secondary window if present.

Be honest about the limit the spike found: codex does NOT expose a per-component
context breakdown (system/tools/MCP/messages token split), so show the real axes
it does give, not a faked breakdown.

Likely surface (for `/plan`): consume `GET /api/agent/context` +
`GET /api/agent/usage`; pure render helpers + jsdom tests; theme with the
existing card/bar styles.

## Decisions (from /plan)

- Frontend-only: the backend `GET /api/agent/context` and `GET /api/agent/usage`
  endpoints (tatr 20260719-212203) already return the data; no backend change.
- Both panels live at the BOTTOM of the sidebar (the slot the sidebar task left):
  a context block (current session) above a weekly-usage block. The sidebar
  becomes a flex column with the session list scrolling (flex:1) and the two
  stat blocks pinned at the foot - claude.ai puts account usage at the bottom.
- Panels are hidden when their data is null (agent off, or no active session),
  so a fresh chat shows just the list; they populate on load, after each reply,
  and after a switch/new.

## Steps

- [x] `web/src/common.ts`: `RateWindow`, `UsageQuota`, `SessionContext` types
      (mirror the backend models).
- [x] `web/src/index.html`: a `.sidebar__foot` holding `#context-panel` and
      `#usage-meter`; make `#session-list` flex so it scrolls above them.
- [x] `web/src/agent-view.ts` (pure helpers exported for jsdom):
      `renderContext(ctx)` (window used bar = input/window %, token breakdown -
      cached/output/reasoning, turns + tool calls; hidden when null) and
      `renderUsage(usage)` (weekly bar = `primary.used_percent`, "N% used",
      "resets in Xd Yh" from `resets_at`, plan type, secondary if present; hidden
      when null). `loadContext()`/`loadUsage()` fetch the endpoints; call them in
      `startAgent`, after each reply, and inside `switchSession`/`newChat`.
- [x] `web/src/style.css`: `.usage-block` / subhead / bars, sidebar flex column
      so the foot stays visible while the list scrolls. Themed.
- [x] `web/src/agent-view.test.ts`: `renderContext` (bar + tokens + turns/tools,
      hidden when null) and `renderUsage` (weekly %, resets label, plan, hidden
      when null).
- [x] LIVE serve smoke: with the agent on, the sidebar shows the weekly-usage
      meter and (after a session is active) the context block; `npm run ci` +
      `ruff`/`mypy`/`pytest` green.

## Definition of Done

- The sidebar shows a weekly-usage meter (used % over the 7-day window, resets
  countdown, plan) and, for the active session, a context block (window used %,
  token breakdown, turn/tool counts). Both hide cleanly when their data is
  absent, update after each turn and on switch/new. No faked per-component
  breakdown. Render side-effect-free for jsdom; jsdom + `npm run ci` + python
  green; serve-verified against the real `$CODEX_HOME`.

## Implementation

- Frontend: `common.ts` `SessionContext`/`RateWindow`/`UsageQuota`; `index.html`
  `.sidebar__foot` (`#context-panel` + `#usage-meter`); `agent-view.ts`
  `renderContext` (window-fill bar + token mix + turns/tools, hidden when null),
  `renderUsage` (weekly bar + used% + resets countdown + plan + secondary, hidden
  when null), `loadContext`/`loadUsage`, `refreshSidebar` (list+context+usage) on
  start / after each reply / on switch/new; `style.css` sidebar flex column so the
  list scrolls and the foot pins, `.usage-block` (+ the `[hidden]` display
  restore). 4 new jsdom tests (37 total).
- Backend correctness fix (found in review): `read_context` now fills
  input/cached from `last_token_usage` (current context occupancy), not the
  cumulative `total_token_usage` which overcounts past the window; output/total
  stay cumulative. Pinned by a test; a real 2-turn session went from a bogus ~23%
  to a truthful 5.6%.
- Live: `/api/agent/usage` -> real weekly window (`plus / 10080 / 1.0%`); real
  multi-turn context fill 14497 -> 5.6%. `npm run ci` + `ruff`/`mypy`/`pytest`.

## Notes

- Spike: tasks/20260719-212152/SPIKE.md.
- Depends on tatr 20260719-212203 (context + usage endpoints - CLOSED).
- Usage refreshes only when a turn runs (label "as of last turn"); do not force a
  refresh turn. Keep render side-effect-free for jsdom; escape host strings.
