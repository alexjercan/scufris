# B5a: reserved orchestrator agent record (synthetic, undeletable, no project)

- PRIORITY: 34
- TAGS: agents, backend
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

First slice of the B5 orchestrator unification (re-cut into B5a-e after an
out-of-context recon showed B5 conflates ~4 architectural changes; user chose
the full split, 2026-07-21). B5a introduces the RESERVED orchestrator as a
synthetic `AgentRecord` surfaced by the store, WITHOUT yet retiring the Agent
protocol or converging chat (those are B5b/B5c/B5d). It makes the orchestrator
a first-class, undeletable agent in the `/api/agents` surface.

## Steps

- [x] Define the reserved id + a synthetic record. `AgentStore`: a fixed
      `ORCHESTRATOR_ID = "orchestrator"`; `get("orchestrator")` returns a
      synthetic `AgentRecord` (name "Orchestrator", project_id "" = no project /
      server cwd, backend from `settings.agent_backend` canonicalized, model
      from `default_model_for`, permission_mode "manual") NOT read from / written
      to agents.json. Its live run-state (session_id/state) is held IN MEMORY.
- [x] `list()` prepends the orchestrator record (always present, first).
- [x] Guards: `delete("orchestrator")` raises `ReservedAgent` -> app.py 403;
      `create` refuses the reserved id/slug -> 422; `update("orchestrator")`
      raises `ReservedAgent` -> 409 (editable config DEFERRED to B5b, where the
      settings-persistence seam is the actual job - the plan's "persist to
      settings" turned out to belong with B5b's unification, not B5a).
- [x] Make `project_id == ""` valid for the orchestrator only (no project
      binding): `_require_agent_project` returns `None` for it and
      `_launch_agent_turn` uses `cwd=None`, so a projectless chat/run runs in the
      server cwd. (This is the seam B5b builds on.)
- [x] Frontend: hide the delete button (card) + the Settings button (detail
      sidebar) for the reserved id; render the project row as "server dir".
- [x] Tests: `get`/`list` include the orchestrator; `delete` -> 403; `create`
      reserved id -> 422; `update` -> 409/ReservedAgent; a projectless
      orchestrator chat streams a turn in the server cwd + persists its session
      in memory. Updated 4 tests that assumed an empty agent list.

## Definition of Done

- The orchestrator is a reserved agent: present in `list()`/`get()`, not in
  agents.json, undeletable (test: `test_orchestrator_reserved_and_undeletable`).
- Its backend/model come from settings and route through `get_backend` on a
  chat turn with `cwd=None` (test: `test_orchestrator_chat_uses_server_cwd`).
- Full check suite green (cmd: `nix develop --command bash -c "ruff check . && mypy . && pytest -q"` + `npm run ci`).
- manual: the orchestrator appears on /agents, opens its page, cannot be deleted.

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (decision 5). Recon in the flow
  transcript (Explore af51a4a614751b102).
- Depends on: F4 (landed). Blocks: B5b.
- Scope guard: B5a does NOT retire the Agent protocol or move multi-session -
  the landing `/api/chat*` + `/api/agent/session/*` endpoints keep working on
  the OLD CodexCliAgent path until B5b/B5c. Two paths coexist during B5a-c
  (temporary; B5d/B5e remove the duplication).
- Persistence decision to pin in /work: an orchestrator config edit reuses the
  settings store's agent_backend/agent_model (no agents.json row), so the
  landing page and the reserved record agree on one source.

## Carried-in note (from B1 review, addressed in B5e)
- `settings-view.ts` BACKENDS still shows raw `app_server`/`mock` ids for the
  process chat agent's `agent_backend` field; reconcile to Codex/Claude in B5e.

## Close-out
- The reserved orchestrator is a SYNTHETIC AgentRecord: `AgentStore` returns it
  from `get`/`list` (never in agents.json), built from `settings.agent_backend`
  (canonicalized) + `default_model_for`. Its run-state (session_id/state) lives
  in memory on the store (`_orch_session_id`/`_orch_state`), updated by
  `mark_running`/`mark_finished` special-cased on the reserved id - so its
  per-agent chat works SINGLE-session in B5a without polluting agents.json.
- Guards: delete->ReservedAgent->403, create-reserved-id->InvalidAgent->422,
  update->ReservedAgent->409. Projectless: `_require_agent_project` returns None
  for it, `_launch_agent_turn` uses `cwd=None` (server cwd).
- KEY scope call: I did NOT wire editable orchestrator config (the plan's
  "persist to settings"). That belongs with B5b, which owns the settings/
  persistence seam as it retires the Agent protocol - doing it in B5a would have
  been a shim. update->409 for now; the landing Settings page still edits
  agent_backend/agent_model (the single source the synthetic record reads).
- Two chat paths coexist during B5a-c (temporary, as planned): the OLD landing
  `/api/chat*` (CodexCliAgent) is untouched; the orchestrator ALSO gains a
  working per-agent chat at `/api/agents/orchestrator/chat` (get_backend path).
  B5d converges them.
- Frontend: no delete/Settings button for the reserved id; project row "server
  dir". Updated 4 tests that assumed an empty list (list is never empty now).
- e2e (real server): `/api/agents` -> [orchestrator]; DELETE -> 403; GET shows
  project="" backend=codex. 274 backend + 168 frontend tests.
