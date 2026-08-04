# Orchestrator permission mode: default to auto + expose in settings

- PRIORITY: 44
- TAGS: feature, agents, backend, config
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

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

- [x] Changed the default of `agent_permission_mode` from `MANUAL` to `AUTO` in
      `config.py` (comment documents the deliberate posture change) and updated the
      `.env.example` sample to `auto` with the same note.
- [x] Confirmed the settings UI already exposes a permission-mode control for the
      orchestrator: the unified settings page (`agent-settings-view.ts`) renders ANY
      agent - orchestrator included - through `agent-fields.ts`'s mode select, and
      PATCH `/api/agents/orchestrator` -> `_update_orchestrator` writes
      `agent_permission_mode` to the settings store. No frontend change needed; the
      round-trip is pinned by the new persistence test.
- [x] Verified the auto sandbox chain: the new FakeBackend assertion proves the
      landing chat passes `permission_mode="auto"`; existing
      `test_codex_backend_permission_mode_flags` maps auto -> danger-full-access, and
      the existing resume-re-sends-sandbox test guards the 20260721-183828 class.
- [x] Tests: orchestrator synthetic record defaults to auto
      (`test_orchestrator_reserved_and_undeletable`); PATCH persists across an app
      restart (`test_orchestrator_permission_mode_defaults_auto_and_edit_persists`);
      the landing turn carries auto (`test_chat_returns_agent_reply`). All written
      first and watched fail red (manual != auto) before the flip.

## Definition of Done

- Fresh install: the orchestrator defaults to `auto`.
  (test: `` `test_orchestrator_reserved_and_undeletable` ``)
- Changing the orchestrator's mode via settings persists and takes effect on the
  next turn.
  (test: `` `test_orchestrator_permission_mode_defaults_auto_and_edit_persists` ``;
  manual: in the running dashboard, the orchestrator settings show auto and a
  change takes effect)

## Notes

- SAFETY: defaulting the LANDING agent to auto is a deliberate posture change - it
  runs shell/writes unattended. Acceptable given the vision (needs write for Bash
  `tatr` after T3 dropped `tatr_new`, and for create project/agent), but call it out
  in the change; the Telegram auth allowlist (T4) becomes the real gate.
- Closes the SPIKE Q4 open question (`tasks/20260722-221359`): orchestrator needs a
  write-capable mode to create tatr tasks via Bash.
- permission_mode values: manual|edit|auto (`enums.py`).

## Implementation (close)

A one-line default flip (`config.py`) plus docs (`.env.example`, CHANGELOG) - the
plan's open question (b) resolved in the code's favor: the unified settings page
already renders the orchestrator's mode select and the PATCH path already persists
it to the settings store, so no frontend change was needed. Three tests written
red-first pin the default, the restart-surviving persistence, and the landing turn
carrying auto; the auto->sandbox mapping and resume-re-send were already pinned by
existing tests, closing the chain end to end.

Notably, the full suite stayed green after the flip - no other test implicitly
assumed the orchestrator's manual default (the pre-checked `manual` assertions all
concern REGULAR agents' record default, which is unchanged).

Self-reflection: checking up front which tests assumed the old default (grep before
flip) meant zero surprise failures; the change landed exactly as scoped.
