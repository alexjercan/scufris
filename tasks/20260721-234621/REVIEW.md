# Review: U3 - unified per-agent settings PAGE component

- TASK: 20260721-234621
- BRANCH: feature/unified-settings-page

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (fresh subagent, no sight of the implementing session;
  ran the web suite itself and re-derived the path-branch correctness, the
  modal-fully-retired sweep, and the escaping)

Clean, well-tested frontend-only slice that respects the U3/U4 boundary
(settings-view.ts NOT retired here). Web `npm run ci` green (prettier + eslint +
154 vitest + webpack build). Verified: the path-branch regex matches only
`/agents/<id>/settings` and runs solely `startAgentSettings` (chat shell hidden);
the component renders for a projectless orchestrator and a project agent, null
panels fall to a dash, save re-loads; `renderSettingsModal`/`AgentDetailActions`
are fully gone; the Health card is genuinely reused (exported) not reimplemented;
all untrusted strings are `escapeHtml`-wrapped.

- [x] R1 (MINOR) web/src/style.css - dead `.agent-modal`/`.agent-modal__card` CSS
  after the modal removal (the `render-rewrite-orphans-its-css` lesson).
  - Response: Fixed. Removed the orphaned `.agent-modal*` rules; `grep agent-modal
    web/src` is now empty.
- [x] R2 (MINOR) agent-detail-view.ts - stale poll comment referencing "the modal
  root".
  - Response: Fixed. Comment now says the sidebar's Settings is a LINK to the
    settings page.
- [x] R4 (NIT) agent-settings-view.ts `statusPanel` - a never-run agent showed a
  bare idle/0/0 instead of "not started".
  - Response: Fixed. `statusPanel` now shows "not started" for an idle,
    sessionless agent (symmetry with the sidebar).
- [ ] R3 (MINOR) the Health card is the GLOBAL `/api/agent/health` (no per-agent
  health endpoint exists), so a claude project agent's page shows a `codex_version`
  row. Deferred (scope) - no per-agent health source; revisit in U4/U5 if a
  per-agent/claude-aware health is wanted.
- [ ] R5 (NIT) `startAgentSettings` hardcodes `writable: true` and surfaces a
  read-only server via a 403-on-save alert; the read-only render branch exists +
  is tested but not wired in prod. Deliberate/scope (the spike's read-only-link
  idea is U4). No change.

### Pending manual DoD (batched at Finish)
- `/agents/<id>/settings` shows the fields + health + panels and editing a field
  saves (user-eyeballed live).
