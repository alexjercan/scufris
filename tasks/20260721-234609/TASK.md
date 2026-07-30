# U2: per-agent settings + panel data endpoints (context/usage/memory/account per agent)

- STATUS: CLOSED
- PRIORITY: 48
- TAGS: agents, backend, spike
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Goal

Give EVERY agent id the settings/panel data the unified settings page needs, so
the frontend fetches ONE shape for any agent (orchestrator or project). A
per-agent analog of the singular `/api/agent/*` endpoints.

- Per-agent effective config (the editable fields + read-only derived bits).
- Per-agent CONTEXT from the agent's own session (`read_context(home,
  session_id)` is already per-session).
- USAGE / MEMORY / ACCOUNT dispatched by the agent's backend + home (account-level
  data for the account backing that agent).
- Decide health: keep the global health card, or a per-agent variant. Reuse the
  existing renders (chat-sidebar renderContext/renderUsage, memory/account).

## Steps (/plan)

- [x] `app.py`: add per-agent panel endpoints that RESOLVE the agent (404 unknown)
      and dispatch by its backend:
      - `GET /api/agents/{id}/usage -> UsageQuota | None` - codex -> `read_usage`;
        claude/mock -> None (no equivalent reader).
      - `GET /api/agents/{id}/memory -> MemoryFootprint` - codex ->
        `read_memory_footprint`; else an empty footprint.
      - `GET /api/agents/{id}/account -> AccountInfo` - `auth_mode`/`model` from the
        agent's EFFECTIVE config; `quota` = codex usage if the agent is codex else
        None.
      A small `_agent_is_codex(agent)` (`canonical_backend(agent.backend)=="codex"`)
      + `resolve_codex_home(settings)` keep the dispatch in one place.
- [x] CONTEXT is NOT a new endpoint: the existing `GET /api/agents/{id}/status`
      already carries per-agent context (turns/tools/tokens/context_window via
      `backend.read_status`) for EVERY backend, so U3 reads the context panel from
      `/status`. The per-agent config (editable fields) is already `GET
      /api/agents/{id}` + `/api/agents/backends`. Note this in the task so U3 wires
      the right sources.
- [x] Tests: for a CODEX agent (a fake `codex_home` with a rollout) each endpoint
      returns real data; for a CLAUDE/MOCK agent usage -> None, memory -> empty,
      account -> quota None; 404 for an unknown id. The orchestrator (codex) also
      resolves these at `/api/agents/orchestrator/*`.
- [x] Full check suite green (ruff + mypy + pytest; no frontend change - U3
      consumes these).

## Definition of Done

- `GET /api/agents/{id}/usage|memory|account` exist, resolve the agent (404
  unknown), and dispatch by backend: real codex-account data for a codex agent,
  None/empty for a non-codex agent (test).
- The orchestrator resolves the same panel endpoints at
  `/api/agents/orchestrator/*` (test).
- Context stays sourced from the existing `/status` (documented; no dup endpoint).
- Full check suite green.

## Notes
- EPIC/umbrella: tasks/20260721-234126. Spike: tasks/20260721-234433/SPIKE.md
  (recommendation U2). Depends on U1 (CLOSED). Backend foundation for U3.
- HONEST SCOPE: usage/memory/account are CODEX-account-level (per codex_home);
  claude has no rollout-usage reader in scufris, so those panels are None/empty
  for a claude agent - correct, not a stub to fill. Context is per-agent for all
  backends via `/status`.
