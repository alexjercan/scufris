# Orchestrator permission mode: default to auto + expose in settings

- STATUS: OPEN
- PRIORITY: 44
- TAGS: feature,agents,backend,config

## Story

As the operator, I want the landing orchestrator to run in `auto` permission mode
by default, and to be able to change its mode from settings, so it can actually do
write work (run commands, create tatr tasks via Bash, create projects/agents)
without me pre-approving each step - especially once it's driven from Telegram.

## Context (grounded)

The orchestrator's mode is NOT hardcoded - it reads `settings.agent_permission_mode`
(`agent_store.py:129`, `_orchestrator_record`), which defaults to
`PermissionMode.MANUAL` (`config.py:117`). That key is ALREADY writable: it is in
`WRITABLE_KEYS` (`settings_store.py:44`) and `AgentConfigUpdate` (`app.py:340`), and
`_update_orchestrator` writes it (`app.py:1033`). So the backend PATCH path
(`PATCH /api/agents/orchestrator` -> `_update_orchestrator`, and
`PATCH /api/agent/config`) already supports changing it. The gaps are: (a) the
DEFAULT is manual; (b) the settings UI may not surface a control for the
orchestrator's permission mode.

## Steps

- [ ] Change the default of `agent_permission_mode` from `MANUAL` to `AUTO` in
      `config.py` (and update the `.env.example` comment).
- [ ] Confirm/expose a permission-mode control for the orchestrator in the settings
      page (the unified settings form - U3/U4/U5). If already present, verify it
      round-trips (PATCH -> reads back on the synthetic record).
- [ ] Verify an orchestrator turn actually runs with the `auto` sandbox after the
      change (codex sandbox mapping; cf. bug 20260721-183828 where resumed turns
      reverted to read-only).
- [ ] Tests: default is auto; PATCH orchestrator permission_mode persists + reflects
      in the synthetic record; a turn maps to the auto sandbox.

## Definition of Done

- Fresh install: the orchestrator defaults to `auto`. (test)
- Changing the orchestrator's mode via settings persists and takes effect on the
  next turn. (test + manual)

## Notes

- SAFETY: defaulting the LANDING agent to auto is a deliberate posture change - it
  runs shell/writes unattended. Acceptable given the vision (needs write for Bash
  `tatr` after T3 dropped `tatr_new`, and for create project/agent), but call it out
  in the change; the Telegram auth allowlist (T4) becomes the real gate.
- Closes the SPIKE Q4 open question (`tasks/20260722-221359`): orchestrator needs a
  write-capable mode to create tatr tasks via Bash.
- permission_mode values: manual|edit|auto (`enums.py`).
