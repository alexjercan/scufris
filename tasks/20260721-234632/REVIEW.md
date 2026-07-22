# Review: U4 - orchestrator-at-root settings symmetry + global sections

- TASK: 20260721-234632
- BRANCH: feature/settings-root-symmetry

## Round 1

- VERDICT: REQUEST_CHANGES
- REVIEWER: out-of-context (fresh subagent; ran the web suite, verified the
  same-component symmetry, no backend/model duplication, the reload thread, and
  the dead-code sweep)

The consolidation is correct - both `/settings` and `/agents/orchestrator/settings`
route through `agentSettingsDeps(ORCHESTRATOR_ID)`, the global sections have no
backend/model duplication, `reload` re-renders after a global mutation, and the
dead composition is fully removed. But the read-only path was left unwired despite
the TASK claiming it closed.

- [x] R1 (MAJOR) agent-settings-view.ts - `agentSettingsDeps` hardcoded
  `writable: true` and never read `config.writable`, so on a read-only server
  (`SCUFRIS_SETTINGS_WRITABLE=0`) the global write controls (System toggles, MCP
  add/remove, tool toggles, profiles) rendered fully interactive and 403 on every
  click. TASK.md claimed "the read-only path gets wired here (config.writable)" -
  it was not.
  - Response: Fixed. Moved `writable` OFF the static `AgentSettingsDeps` and ONTO
    per-load `AgentSettingsData`, derived from the always-fetched
    `/api/agent/config` (`writable: config?.writable ?? true`). `renderAgentSettings`
    now gates BOTH the editable form and the global write sections on
    `data.writable`, and shows a "Read-only server (SCUFRIS_SETTINGS_WRITABLE=0)"
    note + read-only rows when false - so a control can never render live-but-403.
- [x] R2 (MINOR) settings-view.test.ts / agent-settings-view.test.ts - the
  read-only test drove `deps({writable:false})`, a path production never hits.
  - Response: Fixed. Retargeted the read-only tests to `data({writable:false})`
    (the real load path), and assert the "Read-only server" note appears.
- [x] R3 (MINOR) style.css - orphaned `.settings__panels` CSS (its producer
  `renderPanels` was removed).
  - Response: Fixed. Removed the dead `.settings__panels` rule.

## Round 2

- VERDICT: APPROVE
- REVIEWER: in-session (the round-1 out-of-context reviewer verified everything
  except the read-only gap; this round is a focused fix of that one MAJOR + two
  minors. Re-derived the load-bearing claim: on a read-only server
  `config.writable === false -> data.writable === false ->` the form renders
  read-only and NO global write controls render - pinned by
  `hides the global sections on a read-only server even for the orchestrator`.)

Web `npm run ci` green (prettier + eslint + 137 vitest + webpack build). R1/R2/R3
all addressed; nothing else changed.

### Pending manual DoD (batched at Finish)
- `/`, `/settings`, `/agents/<id>`, `/agents/<id>/settings` all load and
  `/settings` == `/agents/orchestrator/settings` in a real browser.
