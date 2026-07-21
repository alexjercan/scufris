# B5: orchestrator as a reserved default agent (multi-session, undeletable, via backend)

- STATUS: OPEN
- PRIORITY: 34
- TAGS: agents,backend


## Goal

Unify the landing orchestrator as a RESERVED default agent: a fixed id (e.g.
`orchestrator`), NOT in agents.json, UNDELETABLE, routed through `get_backend`
like the others (the deferred "decision 4"). It KEEPS the multi-session features
(`/api/agent/session/*`: new/switch/fork/list/delete); project agents are
single-session and do not expose them. The landing page + per-agent chat converge
on the same chat component; the orchestrator page also shows the session switcher.

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (decision 5; recommendation B5). Land AFTER F4 so the chat
  component is already parameterized.
- Depends on: 20260721-112438 (F4).

## Carried-in note (from B1 review)
- The settings page picker (`settings-view.ts` BACKENDS) still shows the raw
  `app_server`/`exec`/`mock` ids for the PROCESS chat agent's `agent_backend`
  Settings field (a separate field from a per-agent record). When B5 unifies the
  orchestrator as an agent, reconcile that picker to the friendly Codex/Claude
  surface (or fold it into the orchestrator's own settings). Tracked so the two
  backend vocabularies do not linger.
