# CRUD control MCP tools for projects and agents (get/update/delete project; update/delete agent)

- STATUS: OPEN
- PRIORITY: 36
- TAGS: feature, agent, mcp, agents, telegram

## Goal

Complete the orchestrator's CRUD reach over projects and agents, so from chat /
Telegram it can not just create and observe but also edit and remove them. This
fills the gaps left by T2 (`tasks/20260722-222722`), which added create/list +
run/message. Each new tool is a thin, bounded wrapper over an existing dashboard
endpoint, following T2's `_api_call` contract exactly (bounded text, never raises,
`_clean_id` on ids).

Scope decision (from the user): the orchestrator edits/removes REGULAR agents
only - it must NOT reconfigure or delete ITSELF. `update_agent`/`delete_agent`
reject `ORCHESTRATOR_ID` with a clear error (delete is already rejected
server-side via `ReservedAgent`; the PATCH endpoint would otherwise allow editing
the orchestrator via the settings store, so the tool must guard it explicitly).

CRUD coverage after this task:
- Projects: create (create_project), read (list_projects + new get_project),
  update (new update_project), delete (new delete_project).
- Agents: create (create_agent), read (list_agents + agent_status), update (new
  update_agent), delete (new delete_agent).

## Steps

- [ ] `get_project(project_id)` -> `GET /api/projects/{id}` (one project's detail).
- [ ] `update_project(project_id, name?, cwd?, language?, description?)` ->
      `PATCH /api/projects/{id}`, body of ONLY provided fields (`ProjectUpdate` is
      `extra="forbid"`, all-optional).
- [ ] `delete_project(project_id)` -> `DELETE /api/projects/{id}`.
- [ ] `update_agent(agent_id, name?, backend?, model?, description?, goal?,
      permission_mode?)` -> `PATCH /api/agents/{id}`, body of ONLY provided fields
      (`AgentUpdate` is `extra="forbid"`). REJECT `ORCHESTRATOR_ID` before the call.
- [ ] `delete_agent(agent_id)` -> `DELETE /api/agents/{id}`. REJECT `ORCHESTRATOR_ID`
      before the call (server also rejects it via `ReservedAgent`, but give a clear
      tool-level message).
- [ ] Guard every id with `_clean_id`; return the updated/deleted record or `error:`
      text. Register all five on the orchestrator-only server; add them to the
      `test_tools_registered` set.
- [ ] Tests (`tests/test_mcp_server.py`, respx): body-only-provided-fields for both
      updates; a `permission_mode` change; delete happy-path; the orchestrator-id
      rejection for update_agent + delete_agent (no HTTP call made); an error path
      (404/422 -> `error:`) and the `_clean_id` guard.
- [ ] CHANGELOG (Added).

## Definition of Done

- All five tools call the correct endpoint with the correct method + body (only
  provided fields on the PATCHes), returning bounded text.
  (test: `` `test_update_project_patches_only_provided_fields` ``,
  `` `test_update_agent_patches_only_provided_fields` ``,
  `` `test_delete_project_calls_endpoint` ``, `` `test_delete_agent_calls_endpoint` ``,
  `` `test_get_project_calls_endpoint` ``)
- `update_agent` and `delete_agent` refuse the orchestrator id with an `error:`
  string and make no HTTP call.
  (test: `` `test_agent_write_tools_reject_orchestrator` ``)
- A non-2xx or bad id yields `error:` text, never an exception. (test: error-path + id-guard)
- All five are registered on the orchestrator-only scufris server.
  (test: `` `test_tools_registered` ``)
- ruff + pytest green; changed source files add zero mypy errors. `nix flake check`
  mypy leg remains pre-existing-red (task 20260720-174021). (cmd: `nix flake check`)

## Notes

- Sibling to T2 (`tasks/20260722-222722`); part of the Telegram frontend spike's
  control-tool set (`tasks/20260722-221359/SPIKE.md` Q2). Useful for the orchestrator
  chat now, independent of the bot (T4/T5).
- Endpoints (all in `scufris/app.py`): GET/PATCH/DELETE `/api/projects/{id}`,
  PATCH/DELETE `/api/agents/{id}`. Models: `ProjectUpdate` (name, cwd, language,
  description; `extra="forbid"`), `AgentUpdate` (name, backend, model, description,
  goal, task_id, permission_mode; `extra="forbid"`). `DeleteResult` = {deleted, current}.
- `ORCHESTRATOR_ID` lives in `scufris.agent_store` (value "orchestrator"); import it
  (lazily, like `_agent_store` does) rather than hardcoding the string.
- permission_mode values: `manual` | `edit` | `auto`.
- Mirror `create_agent`'s input-normalization + the `_api_call` error contract.
