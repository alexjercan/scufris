# A2: AgentBackend interface + codex runner + read-only status + unattended probe

- STATUS: OPEN
- PRIORITY: 26
- TAGS: spike,agents,backend

## Goal

The common backend seam plus the codex implementation and the load-bearing
probe (spike revision 1, decisions 1,4):

- **`AgentBackend` interface**: `run` / `stream` / `status` / `resume`, designed
  so the store, supervisor, dashboard and orchestrator never branch on backend.
  It must hide output format, session resume, MCP config, and permission/sandbox
  model differences (see spike decision 1).
- **codex runner** behind the interface (reuses the existing app_server/exec
  machinery).
- **read-only `agent_status`** `-> {state, last_activity, current_tool, turns,
  tokens, updated_at}` from the agent's codex rollout (reuse sessions.py).
- The **orchestrator** (main chat) also routes through this interface (decision
  4 - the main agent is itself backend-swappable).
- **Probe**: run one long autonomous `codex exec` turn that invokes /flow on a
  scratch project; record unattended behaviour (approval mode, memory growth,
  liveness, failure modes) before A3 commits the UI. Resolves the open question.

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (Q2 rollout-tail; decisions 1,4; the
  "does a long codex-exec /flow turn behave unattended" open question).
- Depends on: 20260720-221929 (A1); built on the A0 supervisor.
- Stepless direction-level task: run /plan before /work.
