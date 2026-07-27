# Remove MCP servers config and Profiles from settings (UI + backend)

- STATUS: CLOSED
- PRIORITY: 2
- TAGS: web,backend,settings

## Problem

The orchestrator settings page shows two features the user does not want:

1. The **"MCP servers"** card - the operator-configurable list of CUSTOM MCP
   servers (id/command/args add + remove forms). Confusing for a single-user,
   single-machine app.
2. The **"Profiles"** card - named config profiles (create/activate/delete).
   No value for a single user on one machine.

Both are removed ENTIRELY: UI and backend (endpoints, request/response models,
config field, on-disk storage, and the profile machinery in the settings
store). The user confirmed a full-stack removal, not a UI-only hide.

## Scope decisions (confirmed with user)

- MCP scope: remove ONLY the **"MCP servers"** management card
  (`renderServerControls`) and the operator `settings.mcp_servers` config. The
  separate **"MCP tools"** section (`renderMcpServers`, fed by the built-in
  in-process `scufris`/`den`/`agent` servers via `mcp_health.servers_for_audience`)
  is KEPT and does NOT depend on `settings.mcp_servers`.
- Backend: remove the endpoints, models, config field and storage too (not a
  UI-only hide).
- The built-in `scufris_mcp_servers` core (agent.py) is KEPT - only the
  operator-declared `settings.mcp_servers` extra-server list is removed. These
  have confusingly similar names; do not touch the `scufris_mcp_servers`
  function or the `_server_override` helper (still used for built-ins).
- Project-capability parsing of a project's own `.mcp.json`/codex
  `[mcp_servers]` tables (`project_capabilities.py`) is UNRELATED and KEPT.

## What is where (map)

Frontend (`web/src/`):
- `settings-view.ts`: `renderServerControls` (165-196), `renderAddServerForm`
  (198-235), `renderProfileSwitcher` (494-557); `SettingsActions` methods
  `addServer`/`removeServer`/`createProfile`/`activateProfile`/`deleteProfile`
  (25-29); imports `McpServerSpec`, `ProfilesResponse` (15-16). KEEP
  `renderMcpServers`/`mcpServerBlock`/`mcpToolCard`/`toolRunner` (MCP tools).
- `agent-settings-view.ts`: mounts at 438 (`renderServerControls`) and 445
  (`renderProfileSwitcher`); `AgentSettingsGlobal.profiles` (52); `profiles`
  fetch (555) + wiring (571-578); `orchestratorGlobalActions` methods (482-493);
  imports (27, 40-41). KEEP `mcpServers` data + `renderMcpServers` calls.
- `common.ts`: `McpServerSpec` (252-257), `mcp_servers` in `AgentConfigUpdate`
  (266), `ProfilesResponse` (356-359). KEEP `McpServerHealth`, `McpServerInfo`?
  (see below). `AgentConfig.mcp_servers` field usage.
- Tests: `settings-view.test.ts` (`renderServerControls` suite 144-197,
  `renderProfileSwitcher` suite 427-461, fake actions 81-85);
  `agent-settings-view.test.ts` (globalSections profiles/actions 151-158,
  assertions 445-446/476-478/495).

Backend (`scufris/`):
- `app.py`: endpoints `POST/DELETE /api/agent/mcp_servers` (887-906),
  `GET/POST/POST/DELETE /api/agent/profiles*` (908-953); models
  `ProfilesResponse`/`ProfileCreate`/`ProfileActivate` (368-380), `McpServerInfo`
  (318), `AgentConfig.mcp_servers` (340), `AgentConfigUpdate.mcp_servers` (364);
  helpers `_validate_mcp_spec` (621), `_apply_mcp_servers` (879); `_agent_config`
  servers list (835-841); imports of profile exceptions + `McpServerSpec`
  (54, 97-102). KEEP `McpServerHealth` (262) and the `/api/agent/mcp` health path.
- `config.py`: `McpServerSpec` class (27), `mcp_servers` field (206), its
  validator (66).
- `agent.py`: the `for spec in settings.mcp_servers` loop (300-305) and any
  helper/constant now used only by it (`BUILTIN_MCP_SERVER_IDS`, `_SERVER_ID_RE`
  - verify no other users). KEEP `scufris_mcp_servers`, `_server_override`.
- `settings_store.py`: collapse the profile machinery - `_profiles`/`_active`,
  `active_profile`/`profile_names`/`create_profile`/`activate`/`delete_profile`,
  `_base_values`/`_reset_to_base`, exceptions `UnknownProfile`/`DuplicateProfile`/
  `InvalidProfileName`/`CannotDeleteProfile`, `PROFILE_NAME_RE`,
  `DEFAULT_PROFILE`. Replace `{active, profiles: {name: {...}}}` persistence with
  a flat overrides shape, with a load-time migration that reads the old
  profile-shaped file (take the active profile's overrides). Remove `mcp_servers`
  from `WRITABLE_KEYS` (48).
- Tests: `test_settings_store.py` (profile tests), `test_app.py` (mcp_servers +
  profiles endpoint tests), `test_agent.py`
  (`test_mcp_overrides_appends_configured_servers_for_any_agent` + collision-skip
  ~125-150, `McpServerSpec` import), `test_config.py` (mcp_servers spec tests).

Open question to resolve during work (not a user fork):
- Whether to drop `AgentConfig.mcp_servers` + `McpServerInfo` entirely (nothing
  in UI consumes it once the card is gone) or keep it reporting only the built-in
  `scufris`. Default: DROP the field and `McpServerInfo` (dead once the card is
  gone) to keep the surface honest; adjust `_agent_config` accordingly.

## Steps

- [x] Sprout worktree from master; `npm ci` in `web/`; run baseline
      `nix develop -c python -m pytest -q` and `npm run ci` green before changes.
- [x] Frontend: delete `renderServerControls`, `renderAddServerForm`,
      `renderProfileSwitcher` from `settings-view.ts` and the five now-dead
      `SettingsActions` methods; fix imports.
- [x] Frontend: in `agent-settings-view.ts` drop the two mounts (438, 445),
      `AgentSettingsGlobal.profiles`, the `profiles` fetch + wiring, and the five
      `orchestratorGlobalActions` methods; fix imports. KEEP all MCP-tools code.
- [x] Frontend: in `common.ts` remove `McpServerSpec`, `ProfilesResponse`, and
      `mcp_servers` from `AgentConfigUpdate`; adjust `AgentConfig` per the
      `mcp_servers`/`McpServerInfo` decision.
- [x] Frontend tests: delete the `renderServerControls` + `renderProfileSwitcher`
      suites and fake-action entries; flip the `agent-settings-view.test.ts`
      assertions so "MCP servers" and "Profiles" are asserted ABSENT for the
      orchestrator while "MCP tools"/"System" stay present.
- [x] Backend `app.py`: delete the mcp_servers + profiles endpoints, the
      `ProfilesResponse`/`ProfileCreate`/`ProfileActivate` models, `_validate_mcp_spec`,
      `_apply_mcp_servers`, `AgentConfigUpdate.mcp_servers`, and the
      `AgentConfig.mcp_servers`/`McpServerInfo` per the decision; fix imports.
      Also update the OpenAPI tag description (154, drop "MCP servers, named
      profiles") and any stale entries in `_route_tags` (167-196).
- [x] Backend `config.py`: remove `McpServerSpec`, the `mcp_servers` field and
      its validator.
- [x] Backend `agent.py`: remove the `settings.mcp_servers` loop and any
      now-unused helper/constant; keep built-in server wiring intact.
- [x] Backend `settings_store.py`: collapse profiles to a flat overrides store
      with a backward-compatible load migration; remove `mcp_servers` from
      `WRITABLE_KEYS`; drop the profile exceptions/regex/constants.
- [x] Backend tests: remove the profiles + configured-mcp_servers tests; add a
      `test_settings_store` migration test proving an old profile-shaped
      `settings.json` still loads its active overrides.
- [x] Full suite green on the branch: `nix develop -c python -m pytest -q` (bare,
      not piped) and `npm run ci` in `web/`. Manually confirm the settings page
      renders with System + MCP tools + Health but no "MCP servers"/"Profiles".

## Definition of Done

1. The orchestrator settings page shows NO "MCP servers" card and NO "Profiles"
   card; "MCP tools", "System", "Health" still render.
   (cmd: `npm run --prefix web test -- agent-settings-view` asserts both absent)
2. No `settings.mcp_servers` config, no profile endpoints/models, no profile
   machinery remain. (cmd: `! grep -rn "mcp_servers\|create_profile\|activate_profile\|ProfilesResponse\|settings.mcp_servers" scufris/app.py scufris/config.py scufris/settings_store.py scufris/agent.py` returns nothing except the KEPT `scufris_mcp_servers`/health/project-capability references)
3. An existing profile-shaped `settings.json` still loads its active overrides
   after the collapse. (test: new migration test in `test_settings_store.py`)
4. Full backend + web suites green.
   (cmd: `nix develop -c python -m pytest -q`; cmd: `cd web && npm run ci`)

## Flow State

- FLOW STEP: DONE
- PLAN STATUS: APPROVED
