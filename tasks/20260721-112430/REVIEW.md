# Review: B2 permission modes (manual|edit|auto)

- TASK: 20260721-112430
- BRANCH: refactor/permission-modes

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Suites (reviewer ran both): backend ruff+mypy clean, 255 passed; frontend
`npm run ci` green, 135 passed. Verified in-session.

No BLOCKER/MAJOR/MINOR/NIT findings. Reviewer verified the load-bearing edges:
- Migration runs BEFORE model_validate, guarded by `"permission_mode" not in
  item` (an existing mode is never clobbered); write_enabled true->edit /
  false-or-absent->manual; no path where a write-enabled agent silently becomes
  read-only (pinned by test_legacy_write_enabled_migrates_to_edit).
- API vs store validation consistent: the API Literal 422s a bad mode; the store
  `normalize_permission_mode` folds unknown->manual (strictly safer). Never
  disagree on valid values.
- Mode->flag map correct (codex read-only/workspace-write/danger-full-access;
  claude default/acceptEdits/bypassPermissions); ClaudeBackend always passes
  --permission-mode; codex --sandbox still applied only on the first turn
  (codex-resume-rejects-sandbox preserved).
- Rename complete (only the intentional migration refs to write_enabled remain);
  default is manual (read-only) at every layer; tests are meaningful.
