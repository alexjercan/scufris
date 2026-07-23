# Review: Orchestrator permission mode - default auto + expose in settings

- TASK: 20260723-001243
- BRANCH: feature/orchestrator-auto-default

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Verified the flip is scoped to the orchestrator only (`agent_permission_mode` is read
in exactly one behavioral place, `agent_store._orchestrator_record`; regular agents'
record/create/model defaults all stay manual, incl. the legacy-migration path); the
auto -> `danger-full-access` sandbox chain incl. resume-re-send is pinned by existing
tests (20260721-183828 class covered); the settings-UI claim is true
(`agent-settings-view.ts` + `agent-fields.ts` render the mode select for the
orchestrator; `_update_orchestrator` writes the settings store; key in
`WRITABLE_KEYS`); the three new tests fail on revert and the persistence test
genuinely restarts the app; the posture change is prominently documented; README /
`web/src/common.ts` "manual default" mentions correctly describe regular agents. One
finding:

- [x] R1.1 (MINOR) scufris/cli.py - the one-shot `scufris chat` orchestrator turn
  passed no `permission_mode`, falling back to the stream default (manual/read-only)
  and silently ignoring both the new default and an explicit
  `SCUFRIS_AGENT_PERMISSION_MODE`. Pre-existing, but the new CHANGELOG claim made it
  inconsistent. Suggested: pass the settings mode, or reword the docs.
  - Response: fixed - `_chat_once` now passes
    `permission_mode=settings.agent_permission_mode.value` (one orchestrator, one
    posture, dashboard or CLI) and its docstring says so; pinned by extending
    `test_chat_subcommand_prints_reply` to assert the recorded kwargs
    (`is_orchestrator=True`, `permission_mode="auto"`). Full suite re-run green.

Open `manual:` DoD item (batched for Finish): in the running dashboard, the
orchestrator settings show mode `auto` and changing it takes effect on the next turn.
