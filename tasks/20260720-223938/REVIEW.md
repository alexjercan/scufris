# Review: A2b ClaudeBackend behind the AgentBackend interface

- TASK: 20260720-223938
- BRANCH: feature/claude-backend

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Check suite (reviewer ran it in the worktree): ruff clean, mypy clean (35
files), `python -m pytest` = 239 passed. Verified independently in-session.

Reviewer confirmed correct: parse_claude_stream skips system/init, maps text ->
StreamTextDelta, tool_use -> StreamTool, result success -> StreamDone(session_id),
non-success -> StreamError (session id captured from the result line, per spec);
stream builds the right argv (--resume only with a session id, cwd forwarded,
stdin closed, line-by-line stdout, kill-on-early-exit); read_status finds the
session by id-glob, counts string user turns while excluding list tool_result
turns, counts tool_use, last assistant text, None on missing; the interface is
UNCHANGED (ClaudeBackend satisfies AgentBackend, read_status keeps its
no-cwd signature); no test depends on a live/authed claude; no regression.

- [x] R1.1 (MINOR) scufris/backends.py stream - `stderr=asyncio.subprocess.PIPE`
  is set but never drained; a chatty claude stderr filling the ~64KB pipe buffer
  could deadlock the turn while we read stdout. Use DEVNULL (or drain it).
  - Response: Fixed. `stderr=asyncio.subprocess.DEVNULL` (stderr is not consumed,
    so DEVNULL removes the deadlock surface entirely) with a comment.
- [x] R1.2 (NIT) parse_claude_stream result branch - a non-success subtype (e.g.
  error_max_turns) with no `result` text falls back to a generic message;
  include the subtype for diagnosability.
  - Response: Fixed. The StreamError detail now falls back to the `subtype`
    before the generic string.
- [ ] R1.3 (NIT) tests - the monkeypatched `_FakeProc.stderr = None` does not
  exercise the stderr path.
  - Response: Acknowledged, not taken - with `stderr=DEVNULL` there is no stderr
    pipe to exercise, so the deadlock surface is gone by construction rather than
    by a test; a live-claude integration test is out of scope (no authed claude
    in CI).

## Round 2

- VERDICT: APPROVE
- REVIEWER: in-session (a one-line robustness fix DEVNULL + a diagnostic string;
  no behavioral change to the parsed events)

Verification: `stderr=DEVNULL` set; the diagnostic subtype fallback added; suite
re-run ruff + mypy clean, 239 passed (unchanged - the fixes touch subprocess
plumbing + an error-string fallback that no existing test pinned). No new
findings.
