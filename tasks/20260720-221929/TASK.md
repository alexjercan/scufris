# A1: AgentStore - agent as a first-class record (agents.json + CRUD)

- PRIORITY: 28
- TAGS: spike, agents
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

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

- [x] Add `scufris/agent_store.py` (named `agent_store` to avoid clashing with
      `agent.py`), mirroring `projects.py`: an `AgentRecord` pydantic model
      `{id, name, project_id, backend, model, goal, task_id, session_id, state,
      write_enabled}` and an `AgentStore` persisting `agents.json` under the
      state dir (atomic write, tolerant load, `settings_writable` gate).
      Exceptions `AgentNotFound`/`InvalidAgent`/`AgentsReadOnly` (no
      `DuplicateAgent`: `_unique_id` dedups so a create never collides).
      `id` = slug from name (reuse the projects `_slugify` + `AGENT_ID_RE`
      pattern) with the same dedup. `state` default `"idle"`; `write_enabled`
      default `False`; `backend`/`model` default from `settings`.
- [x] `create` validates: non-empty name; `project_id` refers to an existing
      project (inject the `ProjectStore` and call `.get`, 422 if unknown);
      `backend` in the known set (`app_server`/`exec`/`mock` for now - A2b adds
      `claude`); the slug yields a valid id. `update` allows name/backend/model/
      goal/task_id/write_enabled (NOT project_id - rebinding an agent's project
      is out of scope). `session_id`/`state` are set by the run machinery (A3),
      not the CRUD API.
- [x] Wire into `app.py`: construct `agents = AgentStore(settings, projects)`
      beside `projects`; add `AgentCreate`/`AgentUpdate` request models
      (`extra="forbid"` on update); endpoints `GET/POST /api/agents`,
      `GET/PATCH/DELETE /api/agents/{agent_id}` mirroring the project routes
      (403 read-only, 404 unknown, 422 invalid; create returns 200). NOTE: the
      per-agent
      `POST /api/agents/{id}/run` + `GET /api/agents/{id}/events` endpoints stay
      deferred to A3/A4; A1 is the record + CRUD only.
- [x] Tests: `tests/test_agent_store.py` (round-trip across a fresh store;
      create validation incl. unknown project_id -> InvalidAgent; dedup ids;
      read-only gate; tolerant load of a corrupt file) and `tests/test_app.py`
      (each new route's own branches: create 200/422/403, get 404,
      patch 404/422/403 + project_id-immutable, delete 404/403 - per
      `test-the-net-new-route-not-the-reused-path`).
- [x] Full check suite green; close-out.

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

## Close-out

What changed:
- New `scufris/agent_store.py`: `AgentRecord` (`{id, name, project_id, backend,
  model, goal, task_id, session_id, state, write_enabled}`) + `AgentStore`
  mirroring `ProjectStore` (agents.json, atomic write, tolerant load,
  settings_writable gate, slug id + dedup). `create` validates non-empty name,
  an existing `project_id` (via the injected `ProjectStore`), and a known
  backend; `update` covers name/backend/model/goal/task_id/write_enabled.
- `scufris/app.py`: `agents = AgentStore(settings, projects)` + `AgentCreate`/
  `AgentUpdate` models + CRUD endpoints `GET/POST /api/agents`,
  `GET/PATCH/DELETE /api/agents/{id}` (403/404/422).
- Tests: `tests/test_agent_store.py` (7: round-trip, unknown-project reject,
  name/backend validation, dedup, read-only gate, corrupt-file tolerance,
  unknown-get) + 3 app-route tests (CRUD, validation incl. patch extra->422 and
  unknown->404, read-only gate).

Decisions:
- Module named `agent_store.py` and model `AgentRecord` to avoid clashing with
  `agent.py`'s `Agent` protocol (both are imported into app.py).
- Store `project_id` (FK), not a cwd snapshot - Project is the single source of
  cwd (the picker); A3 resolves cwd via the project when it runs the agent.
- `backend: str` validated against `KNOWN_BACKENDS` (not the settings Literal) so
  A2b adds "claude" with no persisted-schema change.
- `session_id`/`state` are NOT settable via the CRUD API (they belong to the run
  machinery, A3); the model defaults them (`None`/`"idle"`).

Deferred (moved here from A0's handoff, now pushed to A3): wiring
`CodexCliAgent` to pass a per-agent cwd - only exercised when an agent RUNS (A3),
and it touches the `StreamRunner` fakes, so it lands in one pass there.

Result: 229 tests pass (+10), ruff + mypy clean.

Self-reflection: mechanical mirror of ProjectStore went cleanly; the one real
design call was FK-vs-snapshot for the project link, resolved toward the FK so
Project stays the single source of truth. No surprises.
