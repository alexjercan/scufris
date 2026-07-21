# Review: F1 SPA dynamic routing + agent-detail shell

- TASK: 20260721-112433
- BRANCH: feature/agent-routing

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Suites (reviewer ran both): backend ruff+mypy clean, 258 passed; frontend
`npm run ci` green, 141 passed, webpack emits agent-detail.html. Verified
in-session + an e2e serve.

Reviewer verified every routing edge: the `/agents/{id}` routes sit AFTER all
`/api/...` routes and BEFORE the static mount (so neither is shadowed);
Starlette's non-empty `{id}` segment means `/agents/` falls through to the static
list (tested); `{rest:path}` handles `/settings` and has no real static asset to
swallow; the devServer rewrite order is specific-before-general; 404 when unbuilt;
include_in_schema=False keeps the openapi tag test passing; agentIdFromPath parses
every case (incl. url-encoded) and returns null for the list/non-agent paths;
renderAgentDetail is pure and escapes every host string; the tests assert the two
regressions that matter (list not shadowed, API not the shell).

- [ ] R1.1 (NIT) agent-detail-view.ts stateBadge interpolates escapeHtml(state)
  into a CSS class - harmless (state is a server enum; escapeHtml doesn't
  neutralize class chars anyway). Same cosmetic pattern as agents-view.ts.
  - Response: Left as-is (server enum, not exploitable); if ever wanted, a
    `[a-z-]` slug would be the class-token defense. No change.
