# B1: backend surface cleanup (Codex/Claude only, mock dev-flag, drop exec, per-backend model, labels)

- STATUS: OPEN
- PRIORITY: 50
- TAGS: agents,backend


## Goal

Clean the backend model surface to the two user-facing backends:
- `get_backend("codex")` -> `CodexBackend("app_server")`; `"claude"` ->
  `ClaudeBackend`. DROP `exec` from the user surface. `mock` resolvable and
  listed ONLY when a dev flag is on (`SCUFRIS_ENABLE_MOCK_BACKEND`, default off).
- `KNOWN_BACKENDS` becomes {codex, claude} (+ mock when flagged). Back-compat:
  map legacy persisted `app_server`/`exec` -> `codex` on load.
- Per-backend DEFAULT MODEL: codex -> settings.agent_model; claude -> a claude
  default (fixes the "claude shows gpt-5.5" bug).
- Friendly LABELS in one map so no UI shows raw ids.

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (decision 2; recommendation "Backend model surface").
- Fixes bugs #1 (model) and #3 (raw backend names) at the data layer.
