# Review: codex agent in auto/edit permission mode still runs read-only

- TASK: 20260721-183828
- BRANCH: fix/codex-permission-sandbox

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Fresh subagent, no sight of the implementing session. Ran the full check
suite itself (ruff + mypy "All checks passed!"; pytest EXIT=0) and
re-derived the load-bearing claim (the regression test KeyErrors without the
fix).

- [x] R1.1 (verified) scufris/agent.py:827-833 - `thread/resume` now sends
  `{"threadId", "sandbox"}`; the sandbox is threaded from
  `CodexBackend.stream` via `_codex_sandbox_for(permission_mode)`, not a
  hardcode. `thread/start` still passes it too. Map correct for all three
  modes (manual->read-only, edit->workspace-write, auto->danger-full-access).
- [x] R1.2 (verified) exec-resume vs app-server-resume - the
  `codex-resume-rejects-sandbox` lesson applies only to `codex exec resume
  --sandbox` (exec path, left untouched); the app-server JSON-RPC
  `thread/resume` accepts `sandbox` per the generated `ThreadResumeParams`,
  so there is no rejection risk.
- [x] R1.3 (verified) tests/test_agent.py - the logging-fake regression test
  asserts `resume["params"]["sandbox"] == "workspace-write"`; without the fix
  the subscript raises KeyError, so the test genuinely fails on the bug. The
  fake faithfully mirrors the handshake; the pre-existing streaming test uses
  the start path and is unaffected.
- [x] R1.4 (verified) mid-session mode change - re-sending the current sandbox
  every resume means a permission-mode change takes effect on the next turn;
  desired behavior, a bonus over the old code.
- [x] R1.5 (verified) honesty/scope - close-out narrative matches the diff
  (only functional change is the resume-sandbox line; approval-policy theory
  ruled out by probe, not guessed); appropriately minimal, no refactor.

No open manual DoD items block landing. The task's "manual: an auto codex
agent can create files" was proven live during diagnosis (three probes,
recorded in the close-out).
