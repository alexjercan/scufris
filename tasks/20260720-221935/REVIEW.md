# Review: A2 AgentBackend interface + CodexBackend + status + probe

- TASK: 20260720-221935
- BRANCH: feature/agent-backend

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Check suite (reviewer ran it in the worktree): ruff clean, mypy clean (35
files), `python -m pytest` = 235 passed. Verified independently in-session.

Reviewer verified and confirmed correct: stream delegation forwards
`(prompt, session_id, image_paths, cwd=cwd)` matching both runner signatures and
selects the runner by mode; `read_status` maps `SessionContext` fields correctly
and returns `None` for a falsy/unknown session; `rollout_mtime` guards
falsy-id/None-path/OSError; the `@runtime_checkable` protocol + isinstance is
benign against the async-gen-vs-def method shape (mypy + isinstance both pass);
every test discriminates its mechanism (the forwarding test asserts the exact
arg tuple, the app_server test installs a `fail_exec` that raises if the wrong
runner is used, read_status asserts real rollout-derived values); the deferred
orchestrator rewire is genuinely absent (`/api/chat/stream` still uses the old
supervised path, no `get_backend`/`AgentBackend` reference outside backends.py);
NOTES.md's live-probe claim is internally consistent and no test depends on a
live/authed codex.

- [ ] R1.1 (NIT) scufris/backends.py:127 - `read_status` calls
  `read_transcript(home, session_id)` (default `limit=200`) and reverse-scans the
  materialized list for the last assistant message. Leaner would be a smaller
  limit or a dedicated tail helper.
  - Response: ACCEPTED, not taken. `read_transcript` parses the rollout in full
    regardless of `limit` (it tails to the most-recent `limit` messages in
    memory), so a smaller limit would not avoid the parse, and the in-memory
    reverse-scan of <=200 small dataclasses is negligible next to the file read.
    A dedicated "last assistant message" tail helper in sessions.py is a
    reasonable future cleanup but adds surface for no measurable win here;
    deferring it avoids risk on a green, approved diff. Filed mentally as a
    micro-opt for whenever sessions.py gets a status-oriented reader.
