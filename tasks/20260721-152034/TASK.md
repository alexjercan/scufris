# Bug: switching backend leaves a stale cross-backend session; claude resume fails

- STATUS: OPEN
- PRIORITY: 40
- TAGS: bug,agents,backend

## Story

Reported: chatting with a `claude` agent fails with "chat failed:
errorduringexecution" while `codex` works. Root-caused live (diagnostic
evidence below): sessions are BACKEND-SPECIFIC, but switching an agent's backend
(codex/mock -> claude, the MB1 flow) leaves the OLD backend's `session_id`
persisted. The next claude turn runs `claude -p ... --resume <that-id>`, which
claude cannot find, so its result frame is
`{"subtype":"error_during_execution","is_error":true}` (stderr: "No conversation
found with session ID: ..."). `parse_claude_stream` maps that to a
`StreamError("error_during_execution")`; the F4 chat renders it through markdown,
so `_during_` becomes italic and it displays as "errorduringexecution".

## Evidence (live probes, claude 2.1.193)

- `claude -p "hi" --output-format stream-json --verbose --permission-mode default`
  (no resume) -> `subtype=success`, works. Model default is `claude-opus-4-8[1m]`
  and it is NOT the cause (ClaudeBackend never passes `--model`).
- Two real turns through the actual `ClaudeBackend.stream` with a SAME-backend
  session resume -> both succeed (turn 2 resumes turn 1's UUID, replies "Bye").
- `claude -p ... --resume <unknown-valid-UUID>` ->
  `subtype=error_during_execution, is_error=True`, stderr
  "No conversation found with session ID: <uuid>". This is the exact reported
  failure - a session id claude cannot resume.

## Steps

- [ ] Write a failing regression test FIRST: `AgentStore.update` on a backend
      change must clear the stale `session_id` (and reset run state to idle) -
      the session cannot carry across backends. Assert a codex agent with a
      session id, PATCHed to claude, comes back with `session_id is None`.
- [ ] Fix layer 1 (correct semantics) - `agent_store.py update()`: in the
      `backend_changed` branch (added by MB1 for the model re-default), also set
      `session_id = None` and `state = "idle"`. A backend switch starts a fresh
      conversation.
- [ ] Fix layer 2 (defence in depth) - `backends.py ClaudeBackend.stream`: only
      pass `--resume <id>` when that claude session actually EXISTS on disk
      (`_find_claude_session(...) is not None`); otherwise start fresh. Factor a
      small pure guard so it is unit-testable without running claude. This makes
      claude robust to ANY unresumable session (deleted/expired/cross-backend).
- [ ] Tests: the update-clears-session unit test; a ClaudeBackend test that with
      a session id whose file is ABSENT, the built args omit `--resume` (and
      present -> include it). Keep the existing backend/store suites green.

## Definition of Done

- Switching an agent's backend clears its session id + resets state
  (test: `test_update_backend_change_clears_session`).
- `ClaudeBackend` omits `--resume` for a session that is not on disk and keeps it
  for one that is (test: `test_claude_stream_skips_unresumable_session`).
- Full check suite green (cmd: `nix develop --command bash -c "ruff check . && mypy . && pytest -q"`).
- manual: switch Builder from a used codex/mock agent to claude and chat - it
  replies instead of "error during execution", starting a fresh claude session.

## Notes
- Root cause is orthogonal to the "chat failed" markdown mangling (the underscores
  in `error_during_execution` render as italic). With the bug fixed the message
  will not appear; a separate small F-task can render chat error frames as plain
  text (not markdown) so any future backend error is legible. Noted, not done here.
- Relevant: scufris/agent_store.py (update, MB1 backend_changed branch),
  scufris/backends.py (ClaudeBackend.stream, _find_claude_session), the F4 chat
  onError path (frontend, out of scope).
