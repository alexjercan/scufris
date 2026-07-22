# claude agents show codex-specific health/settings; make the settings page backend-aware

- STATUS: OPEN
- PRIORITY: 60
- TAGS: agents,frontend,backend,bug

## Goal

A claude-backed agent's settings/health page currently surfaces CODEX-specific
things (e.g. the health check reports a `codex_version`, and other settings read
as codex-oriented). For a claude agent the page should show CLAUDE-related facts
instead (claude CLI/SDK version, claude auth/account, claude-appropriate health
checks), dispatched by the agent's effective backend - the same way U2 dispatched
the usage/memory/account panel endpoints by backend.

## Why

User feedback (2026-07-22): "the settings page for 'claude' based agents show
'codex' related things in the health check and other settings, change it to
actually show claude related things". It is a correctness bug: the unified
settings page (U3/U4) shows one backend's system facts regardless of the agent's
backend.

## Notes / scope to pin

- The health endpoint is currently global (`/api/agent/health`) and codex-shaped
  (`codex_version`). It needs to become per-agent and backend-aware (a claude
  agent -> claude version + claude checks; a codex agent -> codex; mock -> mock).
  Relates to U2's `_agent_is_codex` dispatch and U3 review R3 (deferred
  per-agent/claude-aware health).
- Audit every field on the settings page for codex-assumptions (auth_mode labels,
  account/model, sandbox vocabulary) and make each reflect the agent's backend.
- Probably wants a /spike first to map every backend-specific surface on the
  settings page before implementing.
