# U2: per-agent settings + panel data endpoints (context/usage/memory/account per agent)

- STATUS: OPEN
- PRIORITY: 48
- TAGS: agents,backend,spike

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

## Notes
- EPIC/umbrella: tasks/20260721-234126. Spike: tasks/20260721-234433/SPIKE.md
  (recommendation U2). Depends on U1's routing. Backend foundation for U3.
