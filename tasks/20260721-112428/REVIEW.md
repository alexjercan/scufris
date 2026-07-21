# Review: F0 agents UI polish

- TASK: 20260721-112428
- BRANCH: fix/agents-ui-polish

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

CI (reviewer ran `npm run ci` in the worktree): prettier + eslint clean, vitest
135 passed (+2), webpack build compiled. Verified independently in-session.

Reviewer verified the async lifecycle: SSE reattach opens only on an active run
(`isActive(s) && events === null`), so the idle-404-auto-reconnect trap is
genuinely avoided and there is no double-open; the identity-guarded `onerror`
holds; the interval is genuinely bounded (selected + active + not-typing) and the
`typingInForm()` focus-guard prevents the full re-render from wiping the create
form; the `selectedId === id` guards prevent a selection-change race; "not
started" is correctly gated on idle + no session (a finished 0-turn run keeps its
session and still shows real status); escaping/a11y unregressed; both tests are
meaningful.

- [ ] R1.1 (NIT) agents-view.ts - the status `setInterval` is never cleared.
  Acceptable for a page-lifetime SPA (no teardown path in `startAgents`).
  - Response: Accepted, not taken - the page runs for the tab's lifetime and the
    interval body is a cheap boolean check when idle; a teardown hook would be
    dead code today. Revisit if `startAgents` ever gains a lifecycle.
