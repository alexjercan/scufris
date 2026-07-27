# Review: Remove MCP servers config and Profiles from settings

- VERDICT: APPROVE
- ROUND: 1
- REVIEWER: out-of-context (branch diff `master...web/rm-mcp-profiles`)

## Verdict

APPROVE. 0 blockers, 0 major, 0 minor. The removal is comprehensive and
correct end to end.

## What was checked

- Settings-store migration `_overrides_from_persisted`: handles the new flat
  `{overrides}` shape, the legacy `{active, profiles}` shape (active -> default
  fallback), and degenerate/corrupt inputs. Migrated data re-persists flat on
  first write; stale non-writable keys are dropped on load. The removed
  base-value reset logic was only used by profile-switching and is safe to drop
  (overrides are layered once on an env-seeded Settings; no reset needed). The
  on-load `drop_invalid=True` apply does not fire `on_change`, matching the old
  behavior.
- Backend cleanup complete: endpoints, request/response models, exceptions,
  `_validate_mcp_spec`/`_apply_mcp_servers`, and the `Settings.mcp_servers`
  field all gone; no dangling imports.
- Frontend cleanup complete: `renderServerControls`/`renderAddServerForm`/
  `renderProfileSwitcher`, the dead `SettingsActions` methods, the `McpServerSpec`/
  `McpServerInfo`/`ProfilesResponse` types, and the dead CSS all removed.
- KEPT and verified intact: `scufris_mcp_servers`, `_server_override`,
  `_mcp_servers_for_audience`, `/api/agent/mcp` health/catalog, the "MCP tools"
  section.
- Tests: migration tests, the flipped UI absence assertions, and the
  store/API whitelist-sync test are all meaningful.

## Out of scope (noted, not actioned)

- Pre-existing desync: Python `AgentConfigUpdate` carries `claude_model` and
  `agent_permission_mode` that the TypeScript `AgentConfigUpdate` does not. This
  predates this change and is unrelated to the removal; left as-is.
