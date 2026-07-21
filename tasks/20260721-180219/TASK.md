# B5c: orchestrator multi-session in the agent model (switch/fork/list/delete)

- STATUS: OPEN
- PRIORITY: 32
- TAGS: agents,backend

## Goal

Give the reserved orchestrator MULTI-session powers within the unified agent
model (B5b): list/switch/fork/delete its sessions. Project agents stay
single-session (one `session_id`, no switcher). Replace the landing-only
`/api/agent/session/*` endpoints with per-agent session endpoints scoped to the
orchestrator (e.g. `/api/agents/orchestrator/sessions*`), backed by the backend's
own session discovery (codex rollout scan / claude session files) rather than the
in-memory `CodexCliAgent._session_id`.

## Coarse steps (/plan expands)

- [ ] Add a backend capability for session discovery + fork (reuse
      `sessions.list_sessions` for codex; a claude equivalent) behind the
      `AgentBackend` interface (or a per-backend helper).
- [ ] Per-agent session endpoints (list/switch/new/fork/delete) that only the
      orchestrator (multi-session-capable) exposes; project agents 404/hide them.
- [ ] Persist the orchestrator's ACTIVE session id (the single resumable one)
      and let switch/new/fork change it; list is read from disk.
- [ ] Retire the old `/api/agent/sessions`, `/api/agent/session*` endpoints once
      the orchestrator uses the new ones.

## Definition of Done

- The orchestrator can list/switch/fork/delete sessions via the per-agent
  surface; a project agent cannot (test: multi-session only for the reserved id).
- Switching/forking changes the active session; a subsequent turn resumes it.
- Full suite green.
- manual: hold two orchestrator conversations and switch between them.

## Notes
- Depends on: B5b (20260721-180208). Blocks: B5d.
- Session DISCOVERY differs per backend (codex rollouts vs claude session files);
  keep it behind the backend interface so it is not codex-shaped.
