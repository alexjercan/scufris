# B5a: reserved orchestrator agent record (synthetic, undeletable, no project)

- STATUS: OPEN
- PRIORITY: 34
- TAGS: agents,backend

## Story

First slice of the B5 orchestrator unification (re-cut into B5a-e after an
out-of-context recon showed B5 conflates ~4 architectural changes; user chose
the full split, 2026-07-21). B5a introduces the RESERVED orchestrator as a
synthetic `AgentRecord` surfaced by the store, WITHOUT yet retiring the Agent
protocol or converging chat (those are B5b/B5c/B5d). It makes the orchestrator
a first-class, undeletable agent in the `/api/agents` surface.

## Steps

- [ ] Define the reserved id + a synthetic record. `AgentStore`: a fixed
      `ORCHESTRATOR_ID = "orchestrator"`; `get("orchestrator")` returns a
      synthetic `AgentRecord` (name "Orchestrator", project_id "" = no project /
      server cwd, backend from `settings.agent_backend` canonicalized, model
      from `default_model_for`, permission_mode a setting or "manual") that is
      NOT read from / written to agents.json.
- [ ] `list()` prepends the orchestrator record (always present, first).
- [ ] Guards: `delete("orchestrator")` raises a new `ReservedAgent` ->
      app.py 403/409 (undeletable); `create` refuses the reserved id/slug;
      `update("orchestrator", ...)` is allowed for backend/model/description/
      permission_mode but persists to the SETTINGS store (agent_backend/
      agent_model), not agents.json - decide the persistence seam, keep minimal.
- [ ] Make `project_id == ""` valid for the orchestrator only (no project
      binding): `run`/`chat` use `cwd=None` (server cwd) when the agent has no
      project. Confirm `_require_agent_project` / `_launch_agent_turn` handle a
      projectless agent (this is the seam B5b builds on).
- [ ] Frontend: the card + detail page already render any agent; hide/disable
      the delete button for the reserved id and render the project row as "none"
      gracefully.
- [ ] Tests: `get`/`list` include the orchestrator; `delete` -> 403; `create`
      with the reserved id -> 422; `update` changes its backend/model; a
      projectless chat/run uses server cwd.

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
