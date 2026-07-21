# Review: drop codex exec mode + refresh docs

- TASK: 20260721-152746
- BRANCH: feature/drop-exec-mode

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Full suite ran green in the worktree (ruff + mypy 35 files + pytest ~275, no
deadlock). One NIT, addressed.

Verified by the reviewer:
- `CodexBackend` lost its mode + `CodexMode`/`_stream_codex_exec`/`Literal`
  imports; `stream` always uses `_stream_app_server`; no residual refs.
  `get_backend("exec")` still resolves to codex (legacy id, tested).
- The exec RUNNERS (`_run_codex_exec`/`_stream_codex_exec`) are retained and
  remain the CodexCliAgent defaults + non-streaming chat runner; the landing
  path is intact.
- The coercion validator maps a legacy "exec" -> "app_server" (loads, does not
  raise); `AgentConfigUpdate` rejects a NEW "exec" PATCH (422). Distinction
  intentional and sound. The coercion test would fail without the validator.
- `build_agent` has no exec branch; no runtime path can present "exec".
- Tests were updated correctly (repointed to `_stream_app_server`, fixtures
  swapped to app_server/mock, coercion test added) - not deleted-to-green.
- Docs: no stale "exec backend option"; codex/claude/permission/model/mock-flag
  described. Close-out matches the diff.

- [x] R1.1 (NIT) backends.py:16 - the module docstring's historical "CodexBackend
  (exec + app_server)" could read as current behavior.
  - Response: fixed - clarified it as historical and noted the exec mode was
    dropped (20260721-152746).
