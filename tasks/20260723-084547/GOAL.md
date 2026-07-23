# Goal: orchestrator defaults to auto permission mode, editable in settings

- DATE: 20260723
- UMBRELLA TASK: 20260723-084547
- LANDING SCOPE: squash-merge to local `master` via `sprout land`; no push.

## Goal

The landing orchestrator currently defaults to `manual` (read-only) permission
mode, so it cannot do write work (Bash tatr after T3 dropped `tatr_new`, create
projects/agents) without per-step approval - and the mode, while already writable
in the backend, may not be exposed in the settings UI. This run flips the default
to `auto` and makes sure the orchestrator's mode is visible and editable in
settings. Deliberate posture change: the landing agent runs unattended writes;
the Telegram auth allowlist (T4) is the future gate.

## Done means

1. A fresh install's orchestrator record reports `permission_mode: auto`.
   (test: default-mode test on the synthetic record)
2. The orchestrator's mode can be changed from the settings surface and persists
   across an app restart (settings store). (test: PATCH round-trip + fresh-app read)
3. An orchestrator turn actually maps to the auto sandbox on start AND resume.
   (test: sandbox mapping, guarding against the 20260721-183828 regression class)

Overall: ruff + pytest green; changed source files add zero mypy errors (flake
check mypy leg pre-existing-red, task 20260720-174021).

## Tasks

- [x] 20260723-001243 (p44, scufris) Orchestrator permission mode: default auto + expose in settings
      landed dcdc454; 1 review round (APPROVE, 1 MINOR fixed: CLI one-shot chat now
      honours the mode); no frontend change needed (settings UI already exposed it).

## Finish (2026-07-23)

Done-definition verified item by item: (1) fresh-install record reports auto
(`test_orchestrator_reserved_and_undeletable`); (2) settings edit persists across
restart (`test_orchestrator_permission_mode_defaults_auto_and_edit_persists`);
(3) auto sandbox on start AND resume (new FakeBackend/CLI assertions chained with
the pre-existing sandbox-mapping + resume-re-send tests). Master suite green, ruff
clean, conformance (`tatr check --ledger`) clean. Bonus beyond the pin: the CLI
one-shot turn now follows the same posture (review R1.1).

## Manual acceptance (batched for the user at Finish)

- (pending) 20260723-001243: in the running dashboard, the orchestrator settings
  show mode `auto` by default and changing it takes effect on the next turn.
