# Goal: multi-agent orchestrator v1 (background supervisor, backends, dashboard)

- DATE: 20260720
- UMBRELLA TASK: 20260720-224655
- LANDING SCOPE: squash-merge each task to `master` (local default), do NOT
  push (user's call). Standard flow landing.

## Goal

Deliver v1 of the multi-agent orchestrator from tasks/20260720-221748/SPIKE.md
(revision 1): scufris becomes a dashboard that manages multiple agents working
on multiple projects, with the main chat agent acting as a read-only
orchestrator over them. The agent runtime is de-singletoned and runs agents as
background jobs under an in-process supervisor (no request timeout; live SSE via
a per-agent event bus, ADR-001); an `AgentBackend` interface makes codex and
claude (Claude Code headless) interchangeable; an agent is a first-class record;
creating one with a goal launches an autonomous `/flow` run scoped to a project
(with a gated per-agent write opt-in); a dashboard shows live status; and the
orchestrator exposes read-only observation tools.

This flow drives the seeded A-series: A0 (runtime foundation) -> A1
(AgentStore) -> A2 (backend interface + codex + status + probe) -> A2b (claude
runner) -> A3 (create-agent-with-goal e2e) -> A4 (dashboard) -> A5 (orchestrator
observation). Each stepless task is `/plan`'d into Steps when picked up.

## Done means

1. Agents run as background jobs under a supervisor with a concurrency cap; a
   run is not tied to an HTTP request and has no 120s timeout (a per-agent
   budget + heartbeat replaces it). (test: supervisor runs a job past the old
   timeout without being killed)
2. Session/agent state is per-agent cwd, not the single server cwd; two agents
   on two project dirs coexist. (test: per-agent-cwd session listing)
3. Live agent output reaches the browser via SSE relayed from a per-agent event
   bus, independent of who started the run; a dropped subscriber does not kill
   the run. (test: event-bus fan-out + replay)
4. An `AgentBackend` interface (`run`/`stream`/`status`/`resume`) has a codex
   runner and a claude runner behind it; the store/supervisor/dashboard never
   branch on backend. (test: both backends satisfy the interface contract)
5. An agent is a first-class record (`AgentStore`/`agents.json`) with CRUD;
   Project is demoted to the project picker. (test: agent_store_round_trip)
6. Creating an agent with a goal launches an autonomous run (its prompt invokes
   `/flow`) scoped to the project cwd, with a per-agent gated write opt-in, and
   its lifecycle state is tracked. (test: create-with-goal starts a tracked run;
   manual: watch an agent implement a small goal end to end)
7. An Agents dashboard lists agents with live status (list polls; focused agent
   streams via SSE) and hosts agent creation (Projects folded in). (manual: the
   dashboard lists agents, creates one, and shows a live run)
8. The orchestrator has read-only `list_agents` / `agent_status` MCP tools so
   the main chat can answer "what is agent-N working on". (test: the tools
   return live status; manual: ask the orchestrator and get a correct answer)

Overall: the full check suite passes on master (cmd: `nix develop --command
bash -c "ruff check . && mypy . && pytest -q"` plus `npm run ci` in web/), and
`tatr check --ledger LESSONS.md` is clean for this goal's tasks.

## Tasks

Updated as tasks land (one line per land). Order = priority; dependencies noted.

- [x] 20260720-221922 (p30) A0: agent runtime foundation (de-singleton + background supervisor, no request timeout)
      landed 443f8b8; 2 review rounds (out-of-context R1: 2 MAJOR sync-reserve race + unbounded registry, fixed); EventBus + Supervisor, /api/chat/stream relays the bus, no request timeout, cwd seam. 219 tests.
- [x] 20260720-221929 (p28) A1: AgentStore - agent as a first-class record [dep: A0]
      landed 17bad00; 1 review round (out-of-context APPROVE, 1 MINOR dead-code + 2 NIT addressed); AgentRecord + CRUD at /api/agents, project_id FK. 229 tests.
- [x] 20260720-221935 (p26) A2: AgentBackend interface + codex runner + status + probe [dep: A1]
      landed 4d6850a; 1 review round (out-of-context APPROVE, 1 NIT deferred); backends.py (AgentBackend + CodexBackend + MockBackend + get_backend), read-only status via rollout, live probe green + corrected /flow generalization. 235 tests.
- [ ] 20260720-223938 (p25) A2b: claude (Claude Code headless) runner [dep: A2]
- [ ] 20260720-221942 (p24) A3: create-agent-with-goal end to end [dep: A1, A2]
- [ ] 20260720-221951 (p22) A4: Agents dashboard page [dep: A1, A3]
- [ ] 20260720-221957 (p20) A5: orchestrator observation MCP tools [dep: A2]

## Manual acceptance (batched for the user at Finish)

Accumulates `manual:` DoD items as tasks land; presented at Finish.
