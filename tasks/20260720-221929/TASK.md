# A1: AgentStore - agent as a first-class record (agents.json + CRUD)

- STATUS: OPEN
- PRIORITY: 28
- TAGS: spike,agents

## Goal

Make "agent" a first-class entity. Add an `AgentStore` persisting `agents.json`
(mirroring projects.py / settings_store.py: atomic write, tolerant load, gated
by settings_writable) with records `{id, name, project_cwd, backend, model,
goal|task_id, session_id, state, write_enabled}` and CRUD API. `backend` selects
codex vs claude (the common interface, A2/A2b); `write_enabled` is the per-agent,
cwd-scoped write opt-in (decision 3); `state` is the lifecycle
(idle|running|blocked|done|error). Demote Project from a destination page to the
project-picker plumbing behind agent creation (keep projects.py as the data
layer + tatr-tasks endpoint).

## Steps

- [ ] Add `scufris/agent_store.py` (named `agent_store` to avoid clashing with
      `agent.py`), mirroring `projects.py`: an `AgentRecord` pydantic model
      `{id, name, project_id, backend, model, goal, task_id, session_id, state,
      write_enabled}` and an `AgentStore` persisting `agents.json` under the
      state dir (atomic write, tolerant load, `settings_writable` gate).
      Exceptions `AgentNotFound`/`InvalidAgent`/`DuplicateAgent`/`AgentsReadOnly`.
      `id` = slug from name (reuse the projects `_slugify` + `AGENT_ID_RE`
      pattern) with the same dedup. `state` default `"idle"`; `write_enabled`
      default `False`; `backend`/`model` default from `settings`.
- [ ] `create` validates: non-empty name; `project_id` refers to an existing
      project (inject the `ProjectStore` and call `.get`, 422 if unknown);
      `backend` in the known set (`app_server`/`exec`/`mock` for now - A2b adds
      `claude`); the slug yields a valid id. `update` allows name/backend/model/
      goal/task_id/write_enabled (NOT project_id - rebinding an agent's project
      is out of scope). `session_id`/`state` are set by the run machinery (A3),
      not the CRUD API.
- [ ] Wire into `app.py`: construct `agents = AgentStore(settings, projects)`
      beside `projects`; add `AgentCreate`/`AgentUpdate` request models
      (`extra="forbid"` on update); endpoints `GET/POST /api/agents`,
      `GET/PATCH/DELETE /api/agents/{agent_id}` mirroring the project routes
      (403 read-only, 404 unknown, 409 dup, 422 invalid). NOTE: the per-agent
      `POST /api/agents/{id}/run` + `GET /api/agents/{id}/events` endpoints stay
      deferred to A3/A4; A1 is the record + CRUD only.
- [ ] Tests: `tests/test_agent_store.py` (round-trip across a fresh store;
      create validation incl. unknown project_id -> InvalidAgent; dedup ids;
      read-only gate; tolerant load of a corrupt file) and `tests/test_app.py`
      (each new route's own branches: create 201/422/409/403, get 404,
      patch 404/422/403, delete 404/403 - per `test-the-net-new-route-not-the-reused-path`).
- [ ] Full check suite green; close-out.

## Definition of Done

- `AgentStore` round-trips: create/list/get/update/delete survive a fresh store
  over the same dir (test: `agent_store_round_trip`).
- Creating an agent bound to an unknown project is rejected 422
  (test: `create_agent_rejects_unknown_project`).
- CRUD API: `GET/POST /api/agents`, `GET/PATCH/DELETE /api/agents/{id}` with the
  right status codes, gated by `settings_writable` (test: the per-route tests).
- The full suite passes (cmd: `nix develop --command bash -c "ruff check . &&
  mypy . && pytest -q"`).

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (recommendation 1; decisions 1,3).
- Depends on: 20260720-221922 (A0, landed 443f8b8).
- Store `project_id` (FK to a Project), not a `project_cwd` snapshot: the
  Project is the single source of cwd (it becomes the picker), so the agent
  references it and the run machinery (A3) resolves cwd via the project. This is
  what "Project becomes plumbing behind agent creation" means concretely.
- `backend: str` (validated against a known set), not the settings `Literal`, so
  A2b can add `"claude"` without a schema change.
- DEFERRED to A3 (moved from A0's handoff note): wiring `CodexCliAgent` to pass a
  per-agent `cwd` to its runners. That change touches the `StreamRunner` fakes
  (protocol-signature blast radius) and is only exercised once an agent actually
  RUNS, which is A3 - so it lands there in one pass with the run mechanism, not
  in the store-only A1.
