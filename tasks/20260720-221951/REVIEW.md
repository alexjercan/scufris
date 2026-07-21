# Review: A4 Agents dashboard page

- TASK: 20260720-221951
- BRANCH: feature/agents-dashboard

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

CI (reviewer ran `npm run ci` in the worktree): prettier clean, eslint clean,
vitest 133 passed (8 new), webpack build emits `agents.js` + `agents/index.html`.
Verified independently in-session.

Reviewer verified: `renderAgents` is genuinely pure (no side effects; auto-start
only in the thin `agents.ts` entry); every host/user string is escaped before
innerHTML (name/backend via textContent; project/backend/model/goal/writes/
status/last_message escaped; SSE frames via textContent); the hostile-string
test is real; the create form submits the right shape; the select-race guard is
present (`if selectedId === id`); the EventSource is closed on select/deselect/
disappeared-selection/error/reopen; the DoD tests are meaningful; webpack
multipage wiring is correct; the fold-as-picker (Projects kept) matches spec; no
test depends on a live server.

- [x] R1.1 (MINOR) agents-view.ts `openEvents` - `source.onerror` closed the
  module-level `events` var, so a stale source's late onerror could close a
  newer stream. Guard on identity.
  - Response: Fixed. `source.onerror` now closes only `if (events === source)`,
    matching the id-guard pattern used in `onmessage`.
- [ ] R1.2 (NIT) agents-view.ts `stateBadge` - `escapeHtml(state)` into a
  className is defensive theater (set via `.className`, never innerHTML; `state`
  is a backend enum). Harmless.
  - Response: Left as-is - harmless and cheap; removing it would only save a call
    while a future non-enum state value would then land unescaped in a class.

## Round 2

- VERDICT: APPROVE
- REVIEWER: in-session (a one-line identity guard on a stale EventSource's
  onerror; no render/behavior change)

Verification: `source.onerror` now guards `events === source`. `npm run ci`
re-run: prettier + eslint clean, vitest 133 passed, webpack build green. No new
findings.
