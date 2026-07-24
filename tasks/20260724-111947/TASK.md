# Session ownership index + multi-session history; drive the switcher from it

- STATUS: CLOSED
- PRIORITY: 62
- TAGS: agents, sessions, backend, bug

## Story

As an operator running the orchestrator plus sub-agents, I want the
orchestrator's chat switcher to show only the orchestrator's own chats, so that
a `codex` sub-agent bound to the server directory stops leaking its conversation
into the orchestrator's session list. The fix records session ownership in
scufris's own registry instead of inferring it from a disk scan, which also
gives the claude and opencode backends multi-session listing for free.

See `tasks/20260724-111947/DECISION.md` for the ownership model and the
forward-only (no-backfill) choice, and `tasks/20260724-111839/SPIKE.md` for the
wider design.

## Steps

- [x] Grow the `SessionRegistry` entry shape (`agent_store.py`) from
      `{backend, session_id}` to
      `{backend, session_id, sessions: [id,...], parent_agent_id: str|None}`.
      Make `_load` tolerate the legacy shape (a `{backend, session_id}` entry
      loads as `sessions=[session_id]`, `parent_agent_id=None`); keep `_persist`
      writing the new shape. `parent_agent_id` is reserved for part 3 - store it,
      do not use it here.
- [x] Add registry methods: `sessions_for(agent_id, backend) -> list[str]` (the
      history under that backend, `[]` on backend mismatch/absent);
      `add(agent_id, backend, session_id)` (set current + append to history,
      dedup; reset history if the backend differs from the stored entry);
      `set_current(agent_id, backend, session_id|None)` (switch/clear current
      WITHOUT dropping history - append the id if unseen; reset to a fresh entry
      when the backend differs); `remove(agent_id, backend, session_id)` (drop
      one id from history, clearing current if it was that id). Keep `get`,
      `has`, `clear` as they are.
- [x] Repoint `AgentStore` call sites: `mark_finished` uses `add` (was `set`);
      `set_orchestrator_session(id)` uses `set_current` (was `set`/`clear`) so
      "new chat" (`None`) preserves history; the legacy-migration in `_load` and
      the backend-switch/delete-agent paths keep using `set`/`clear`
      respectively. Add `orchestrator_sessions() -> list[str]`
      (`sessions_for(ORCHESTRATOR_ID, orch_backend)`) and
      `forget_orchestrator_session(session_id)` (`remove` under orch backend).
- [x] Rewrite `GET /api/agent/sessions` (`app.py`) to build the list from
      `agents.orchestrator_sessions()`: resolve the orchestrator's backend via
      `get_backend(agents.get(ORCHESTRATOR_ID).backend)`, hydrate each id with a
      new `backends.session_info(backend, settings, sid) -> SessionInfo | None`
      helper (placed in `backends.py`, next to the backend protocol, rather than
      `app.py`) - title = first `user` message from `backend.read_transcript`,
      `started_at` = that message's `ts`, `updated_at` =
      `backend.read_status(...).updated_at` as a UTC datetime; `git_branch`/`cwd`
      left None; drop unhydratable ids, sort newest-first by activity. `current`
      stays `agents.orchestrator_session_id()`. Remove the `list_sessions` import
      from `app.py`.
- [x] Make `DELETE /api/agent/session/{session_id}` also call
      `agents.forget_orchestrator_session(session_id)` so a deleted session
      leaves the switcher list (in addition to unlinking the rollout and
      resetting current when it was active).
- [x] Tests (registry, `tests/test_agent_store.py`): `add` accumulates multiple
      ids under one agent; `set_current(None)` preserves history; `set_current`
      to an unseen id appends it; `remove` drops one id; a backend switch clears
      history; a legacy `{backend, session_id}` entry loads as a one-element
      history.
- [x] Test (the leak regression, `tests/test_app.py`): write two codex rollouts
      in the same home/cwd - one the orchestrator's, one a sub-agent's - register
      only the orchestrator's via `set_orchestrator_session`, and assert
      `GET /api/agent/sessions` returns only the orchestrator's id (the
      sub-agent's is absent). This must go red on master.
- [x] Test (multi-session): two orchestrator sessions both appear, newest first;
      and update the existing `test_delete_session_removes_and_resets_current` /
      `test_sessions_lists_and_reports_current` expectations if the hydration
      path changes their shape.
- [x] Append a NOTES.md design/fix record (what changed, the leak diagnosis, the
      forward-only tradeoff) per AGENTS.md.

## Definition of Done

- A sub-agent's codex session in the same home/cwd does NOT appear in
  `GET /api/agent/sessions` (test: `test_orchestrator_switcher_excludes_subagent_sessions`).
- The orchestrator's switcher lists multiple of its own sessions, newest first,
  driven by the registry (test: `test_orchestrator_switcher_lists_registry_history`).
- "New chat" keeps prior sessions in the history; only `current` resets
  (test: `test_new_chat_preserves_session_history`).
- Deleting a session removes it from both disk and the switcher list
  (test: `test_delete_session_removes_and_resets_current`).
- A legacy `sessions.json` entry still loads and lists
  (test: `test_legacy_session_entry_loads_as_single_history`).
- The switcher no longer calls the disk-scan lister
  (cmd: `grep -n list_sessions scufris/app.py` prints nothing).
- Full QA gate is green (cmd: `nix flake check`).

## Notes

- Spike: tasks/20260724-111839/SPIKE.md (part 1). Decision:
  tasks/20260724-111947/DECISION.md.
- Relevant files: `scufris/agent_store.py` (`SessionRegistry`, `AgentStore`),
  `scufris/app.py` (`get_sessions` ~L1650, `delete_agent_session` ~L1724,
  `SessionsResponse`/`SessionInfo`), `scufris/sessions.py` (`list_sessions`,
  `_find_rollout`, `SessionInfo`), `scufris/backends.py`
  (`read_transcript`/`read_status` per backend).
- `list_sessions` stays for `health.py:256`'s diagnostic count (a number; the
  leak does not matter there). Do not delete it.
- SUPERSEDES task `20260720-020345` ("List app_server sessions... originator
  fix"): that narrow originator patch is obsoleted by recording ownership. Flag
  it for close when this lands.
- Non-codex session DELETE still only unlinks a codex rollout (pre-existing);
  the registry removal added here is backend-agnostic. Backend-specific delete
  is out of scope.
- Perf: hydrating each id via `read_transcript`/`read_status` reads more than a
  head-only scan; fine for a short list, revisit if a large history is slow (see
  DECISION.md Consequences).
- Parts 2 (tatr 20260724-111955) and 3 (tatr 20260724-111959) depend on this.

## Outcome (CLOSED)

Shipped on `fix/session-ownership-index`. The switcher is now driven by the
ownership registry, so a sub-agent's codex session in the same home/cwd no
longer leaks into the orchestrator's list; claude/opencode get multi-session
listing for free. See `NOTES.md` for the design/fix record and `DECISION.md` for
the ownership model + forward-only choice.

- All Steps landed. Helper placed in `backends.py` (`session_info`), not
  `app.py`, so it sits with the backend protocol (step amended to match).
- Every DoD proof passes; the leak repro
  (`test_orchestrator_switcher_excludes_subagent_sessions`) was written first,
  went red on master (`['sub-sess','orch-sess'] == ['orch-sess']`), and an A/B
  sabotage re-reddened it. `nix flake check` green.
- Follow-up left for the caller: task `20260720-020345` (the narrow originator
  patch) is superseded and should be closed when this merges.
- Diagnosed one snag: `AgentStore.list` shadows builtin `list` in class-scope
  annotations (mypy); worked around with a module-level `SessionIdList` alias.
