# Route orchestrator session endpoints through the backend (transcript/context/fork/delete)

- STATUS: CLOSED
- PRIORITY: 42
- TAGS: agents, sessions, backend

## Story

The orchestrator's session LIST is backend-agnostic (part 1), but
`get_session_transcript`, `get_context`, `fork_session`, and
`delete_agent_session` still hardcode `resolve_codex_home` + codex module
readers, so a claude/opencode orchestrator lists sessions it cannot open,
context and fork are codex-shaped, and delete only unlinks a codex rollout.
Route all four through the orchestrator's backend, adding the two capabilities
the `AgentBackend` protocol lacks, so `app.py` never branches on backend.

Umbrella: 20260724-124151. Follows parts 1 (20260724-111947) and 2
(20260724-111955), both LANDED.

## Steps

- [x] Add `read_context(self, settings, session_id) -> SessionContext | None` to
      the `AgentBackend` protocol (`backends.py`). CodexBackend delegates to
      `sessions.read_context` (the rich rollout reader - keeps cached/reasoning/
      total/window). Claude and Opencode map their `read_status` into a
      `SessionContext` via a shared helper `_context_from_status(status)`
      (session_id, context_window, input/output tokens, turn/tool counts; the
      codex-only cached/reasoning/total stay 0). MockBackend returns None.
- [x] Add `delete_session(self, settings, session_id) -> bool` to the protocol.
      Codex: `sessions.delete_session(resolve_codex_home(settings), sid)` (unlink
      rollout). Claude: unlink the file `_find_claude_session` locates. Opencode:
      add `OpencodeClient.delete_session` (`DELETE /session/{id}`, tolerant of a
      404/network error -> False) and call it. Mock: return False (nothing on
      disk). All never raise.
- [x] `app.py` `get_session_transcript` (~L1733): replace
      `read_transcript(resolve_codex_home(...), sid)` with
      `get_backend(agents.get(ORCHESTRATOR_ID).backend).read_transcript(settings, sid)`.
- [x] `app.py` `get_context` (~L1724): replace
      `read_context(resolve_codex_home(...), current)` with
      `get_backend(...).read_context(settings, current)`.
- [x] `app.py` `fork_session` (~L1697): build the seed from
      `get_backend(...).read_transcript(settings, source_id)` instead of the codex
      home reader; the rest (format_fork_seed + `_launch_agent_turn`) is already
      backend-agnostic.
- [x] `app.py` `delete_agent_session` (~L1741): call
      `get_backend(...).delete_session(settings, sid)` for the `deleted` bool
      (was `sessions.delete_session(resolve_codex_home...)`) and keep the existing
      `forget_orchestrator_session`. Drop the now-unused codex-only imports if
      nothing else uses them.
- [x] Tests: claude-backed orchestrator transcript/context/fork/delete route
      through ClaudeBackend (use `_write_claude_session` + a claude-backed
      settings); opencode client `delete_session` issues `DELETE /session/{id}`;
      codex paths still work (existing tests unchanged). Confirm each new backend
      method on all four backends.
- [x] NOTES.md design/fix record. Add a DECISION.md if the protocol extension is
      deemed load-bearing (it is - two new capability methods on the seam).

## Definition of Done

- A claude-backed orchestrator re-renders a session transcript via the backend
  (test: `test_orchestrator_transcript_uses_backend`).
- A claude-backed orchestrator returns a `SessionContext` from the backend, and
  codex stays rich (test: `test_orchestrator_context_uses_backend`; existing
  `test_context_endpoint_returns_snapshot` still passes).
- Fork seeds from the backend's transcript
  (test: `test_orchestrator_fork_uses_backend_transcript`).
- Delete calls the backend's provider delete + registry forget
  (test: `test_orchestrator_delete_uses_backend`; existing
  `test_delete_session_removes_and_resets_current` still passes).
- `OpencodeClient.delete_session` issues `DELETE /session/{id}`
  (test: `test_delete_session_issues_delete`).
- The four orchestrator session endpoints no longer call `resolve_codex_home`
  (cmd: `grep -n "resolve_codex_home" scufris/app.py` shows only usage/memory/
  account endpoints).
- Full QA gate green (cmd: `nix flake check`).

## Notes

- Umbrella: tasks/20260724-124151/GOAL.md. Spike: tasks/20260724-111839/SPIKE.md.
- Relevant files: `scufris/backends.py` (AgentBackend protocol + Codex/Claude/
  OpenCode/Mock), `scufris/opencode_client.py` (`delete_session`),
  `scufris/app.py` (the four endpoints), `scufris/sessions.py`
  (`read_context`/`delete_session`, still used by codex + health).
- `resolve_codex_home` stays imported for the codex-specific usage/memory/account
  endpoints (`get_usage`/`get_memory`/`get_account`) - those are genuinely
  codex account reads, out of scope here.
- `read_context`'s codex path keeps the rich token breakdown; the mapped
  claude/opencode path reports what `read_status` has (window 0 - those backends
  do not expose a context window), which is honest, not lossy-by-bug.
- Implementation note: `delete_session` is async on the backend protocol so
  opencode can use `OpencodeClient.delete_session` without nesting event loops
  inside FastAPI's async delete route; codex/claude still perform local unlink
  cleanup behind async wrappers.
- Difficulty hit: app tests initially failed because they read the real settings
  override store and switched away from the test backend. Fixed by giving the new
  app tests an isolated `state_dir`, matching the existing test lesson.
- Self-reflection: the first opencode delete draft put HTTP DELETE directly in
  `OpenCodeBackend`, which worked but bypassed the client boundary named by the
  task. Next time, line up tests with the intended ownership boundary before
  running the first focused slice.

## Outcome (CLOSED)

The four orchestrator session endpoints (transcript / context / fork / delete)
now route through `get_backend(orchestrator.backend)` instead of the codex home,
so they work for codex, claude, and opencode. `AgentBackend` gained
`read_context` (sync; codex rich, others mapped from status) and async
`delete_session` (codex unlink / claude file unlink / opencode `DELETE` via
`OpencodeClient`; mock no-op). See DECISION.md for the sync-reads/async-delete
choice and NOTES.md for the design + difficulties.

- All Steps landed; every DoD proof passes; 440 pytest + ruff + mypy +
  `nix flake check` green. DoD grep confirms `resolve_codex_home` is gone from
  the session endpoints (only usage/memory/account remain).
- A/B: reverting the transcript endpoint to a codex-home read turned
  `test_orchestrator_transcript_uses_backend` red; restoring it green.
