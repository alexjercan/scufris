# Review: Match sub-agent Tools section to orchestrator tool cards

Round 1 - out-of-context reviewer (Explore agent) against the uncommitted diff
on `fix/subagent-tools-cards`.

## Scope

3 files: `web/src/settings-view.ts` (export `toolCard`),
`web/src/agent-settings-view.ts` (`agentToolsPanel` refactor), and
`web/src/agent-settings-view.test.ts` (structure assertions).

## Findings

- Correctness: PASS. `agentToolsPanel` now renders a `tool-grid` of shared
  `toolCard(t)` inside the existing `settings__card` + `Tools (N)` section;
  empty-state "none" note preserved via `panel(...)`.
- Read-only guarantee: PASS, structurally enforced. The toggle and "Try it"
  runner are added only by `toolControlCard`/`renderToolControls`, never by the
  bare `toolCard` the sub-agent path uses. A checkbox/runner cannot appear.
- Orchestrator unchanged: PASS. Zero changes to `renderToolControls`/
  `toolControlCard`; only `toolCard` visibility (private -> exported) changed.
- Imports: PASS. `el` and `escapeHtml` still used elsewhere in the file; no
  dangling imports (eslint in CI also green).
- Test quality: PASS. Asserts `.tool-grid`/`.tool-card`/`.tool-card__name`/
  `__server`/`__args`, plus no `input[type=checkbox]` and no `.tool-runner`.
  Empty-state and orchestrator-exclusion tests retained.
- Drift risk: PASS. Both surfaces render the one `toolCard`, so future card
  changes hit both equally.

## Verification

`nix develop --command bash -c 'cd web && npm run ci'` green (format:check, lint,
186 vitest tests, webpack build).

- VERDICT: APPROVE
