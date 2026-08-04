# claude agents show codex-specific health/settings; make the settings page backend-aware

- PRIORITY: 60
- TAGS: agents, frontend, backend, bug
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

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

## Spike findings (mapped 2026-07-22, folded in - no separate SPIKE task)

The recon nailed down exactly one root cause + the full backend-specific surface:

- ROOT CAUSE: the settings page (`agent-settings-view.ts:431`) fetches the GLOBAL
  `/api/agent/health` for EVERY agent. That endpoint calls `agent_health(settings)`
  which probes `settings.agent_backend` (the orchestrator/server backend, usually
  codex). So a claude project agent shows the ORCHESTRATOR's codex health.
- `health.agent_health()` ALREADY branches the CLI probe by backend (codex CLI +
  auth / claude CLI / mock nothing) - it just reads the global backend, not the
  agent's. So the fix is to parametrize it by an effective backend, not to add new
  probe logic.
- `AgentHealth` has a codex-specific field `codex_version` (health.py:40); the
  frontend (`settings-view.ts:294`) reads `health.codex_version`. This must become
  backend-neutral.
- The per-agent `/api/agents/{id}/health` does not exist yet; `_require_agent`
  (app.py:988 -> `agents.get`) DOES resolve the orchestrator (U1), so one per-agent
  endpoint serves project agents AND the orchestrator settings page.
- Other surfaces audited: `permission_mode` (manual/edit/auto) is already
  backend-neutral (mapped per backend in `backends.py`); `agent.model` on the
  account panel is already per-agent. The account panel's `auth_mode`
  (`AccountInfo.auth_mode` from global `settings.agent_auth_mode`,
  chatgpt|api_key) is codex/ChatGPT-centric and the app does not model claude's
  auth - see the DoD boundary below.

## Steps (/plan)

- [ ] `health.py`: parametrize `agent_health(settings, *, backend: str | None =
      None)` - probe the given backend's CLI (default = `settings.agent_backend`,
      preserving the orchestrator/global behavior). Generalize `AgentHealth`:
      REPLACE `codex_version` with backend-neutral `backend: str` (the effective
      backend) + `backend_version: str | None` (the CLI `--version` output, set in
      BOTH the codex and claude branches; None for mock/missing). Keep everything
      else (session summary, MCP, web assets) as-is - they are server-global and
      correct for any agent.
- [ ] `app.py`: add `GET /api/agents/{id}/health` -> `_require_agent` (404) then
      `agent_health(settings, backend=agent.backend)`. Keep the global
      `/api/agent/health` (system/orchestrator health) unchanged for compatibility.
- [ ] `common.ts`: update the `AgentHealth` type (`backend: string;
      backend_version: string | null;` replacing `codex_version`).
      `settings-view.ts renderHealthCard`: show `backend_version` generically
      (was `codex_version`); optionally surface the backend label.
- [ ] `agent-settings-view.ts`: fetch the per-agent `/api/agents/${enc}/health`
      (was the global `/api/agent/health`) so each agent's Health card reflects ITS
      backend - including the orchestrator settings page (id `orchestrator`).
- [ ] Tests: backend - `agent_health(settings, backend="claude")` probes claude
      even when `settings.agent_backend="codex"`; `GET /api/agents/{id}/health`
      returns claude checks for a claude agent and codex for a codex agent, and
      resolves the orchestrator; a bad id 404s. Frontend - `renderHealthCard`
      renders `backend_version` (not a codex-only field); the settings page hits
      the per-agent health URL. Full suite (`ruff`/`mypy`/`pytest`, web `npm run
      ci`) green.

## Definition of Done

- A claude agent's settings page Health card probes the CLAUDE cli (not codex) and
  shows the claude version; a codex agent shows codex; mock shows neither. Proven
  by a backend test on `GET /api/agents/{id}/health` per backend + the frontend
  fetching the per-agent URL.
- `AgentHealth` carries no codex-specific field name; `backend`/`backend_version`
  are backend-neutral and the frontend renders them generically.
- The orchestrator's settings Health still works (per-agent endpoint with the
  orchestrator id === the old global behavior for the orchestrator's backend).
- Full backend + web suites green.
- BOUNDARY (explicitly out of scope, noted not fixed): the account panel's
  `auth_mode` stays the global `settings.agent_auth_mode` (chatgpt|api_key). The
  app does not model claude's auth, so inventing a claude auth mode is deferred;
  the account panel's `model` is already per-agent and correct. If the user wants
  claude auth surfaced, that is a follow-up.
