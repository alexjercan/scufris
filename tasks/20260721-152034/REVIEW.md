# Review: backend-switch clears session (claude resume bug)

- TASK: 20260721-152034
- BRANCH: fix/backend-switch-clears-session

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Full suite ran green in the worktree (ruff + mypy 35 files + pytest ~1s, no
deadlock). Zero findings.

Verified by the reviewer:
- Layer 1 (agent_store.py): on `backend_changed` clears `session_id` + resets
  `state` to idle. `backend_changed` compares canonical-vs-canonical (incoming
  folded via `canonical_backend`, stored backend is canonical on create + on
  load-migration), so a legacy `app_server`->`codex` PATCH is NOT a change and a
  valid session is not wiped. Same-backend/description-only update leaves the
  session untouched. Independent of the MB1 model re-default in the same branch.
- Layer 2 (backends.py): `_claude_stream_args` is pure and adds `--resume` only
  when `_find_claude_session` finds the session on disk; the non-resume path
  builds identical args and still passes cwd. Benefits codex too (a stale
  claude/mock id switched to codex is cleared by layer 1).
- Tests genuinely gate each layer (would fail without the fix); the fixed
  existing test seeds a real on-disk session and still exercises the
  --resume-present path; no other tests needed updating.
- Close-out's live evidence matches the code.

No BLOCKER/MAJOR/MINOR/NIT issues. APPROVE.
