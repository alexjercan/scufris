# CRUD control MCP tools for projects and agents (get/update/delete project; update/delete agent)

- PRIORITY: 36
- TAGS: feature, agent, mcp, agents, telegram
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

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

- [x] `get_project(project_id)` -> `GET /api/projects/{id}` (one project's detail).
- [x] `update_project(project_id, name?, cwd?, language?, description?)` ->
      `PATCH /api/projects/{id}`, body of ONLY provided fields via the new `_provided`
      helper (`ProjectUpdate` is `extra="forbid"`); refuses an empty update.
- [x] `delete_project(project_id)` -> `DELETE /api/projects/{id}`.
- [x] `update_agent(agent_id, name?, backend?, model?, description?, goal?,
      permission_mode?)` -> `PATCH /api/agents/{id}`, body of ONLY provided fields.
      Rejects `ORCHESTRATOR_ID` before the call (new `_reject_orchestrator` helper).
- [x] `delete_agent(agent_id)` -> `DELETE /api/agents/{id}`; also rejects
      `ORCHESTRATOR_ID` tool-side (server rejects it too via `ReservedAgent`).
- [x] Guard every id with `_clean_id`; return the record/result or `error:` text.
      Registered all five on the orchestrator-only server; added them to
      `test_tools_registered`. Updated the module docstring to describe the full CRUD.
- [x] Tests (`tests/test_mcp_server.py`, respx): body-only-provided-fields for both
      PATCHes, a `permission_mode`+`backend` change, both deletes, the get, the
      orchestrator-id rejection (no HTTP call), a 404 error path, empty-update guards,
      and the `_clean_id` guard across the CRUD tools.
- [x] CHANGELOG (Added).

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

## Implementation (close)

Added five CRUD control tools plus two helpers (`_provided` builds a PATCH body of
only the non-None fields since `ProjectUpdate`/`AgentUpdate` are `extra="forbid"`;
`_reject_orchestrator` refuses the reserved id). All follow T2's `_api_call`
contract (bounded text, never raises) and `_clean_id` guard. The write tools on
agents refuse `ORCHESTRATOR_ID` before any HTTP call, per the user's "regular
agents only" scope - the orchestrator edits itself via settings, not here.

Grounding that kept it small: the endpoints already existed (GET/PATCH/DELETE
`/api/projects/{id}`, PATCH/DELETE `/api/agents/{id}`); this task is pure
tool-surface wiring. `delete_agent` is doubly safe (server also raises
`ReservedAgent` for the orchestrator), but the tool-level guard gives a clearer
message and avoids a needless HTTP round-trip.

Verification: ruff + full pytest green (358 tests, +10 new CRUD tests); mypy clean
on the changed source file. `nix flake check` mypy leg remains pre-existing-red
(task 20260720-174021).

Self-reflection: applied the prior retros' lessons up front - grepped the tool
registration set and updated it in the same pass (protocol-doubles lesson), and the
orchestrator-reject is pinned by a test that makes NO http call (so it can fail if
the guard is removed). No review NITs anticipated on id-handling since `_clean_id`
and the error contract were reused verbatim from T2.
