# Spike: scufris as a multi-agent orchestrator - what is an agent instance, and how do we observe it?

- DATE: 20260720-221748
- STATUS: RECOMMENDED   # revision 1 - decisions locked below
- TAGS: spike, agents, orchestrator, dashboard

## Question

We want scufris to become a **dashboard for managing multiple agents working on
multiple projects**, with the existing chat page (the "main" agent) acting as an
**orchestrator** over them: I can say "implement feature X in project Y" and it
spins up an agent scoped to project Y with goal X (which then runs the `/flow`
skill), and I can ask the orchestrator "what is agent-2 working on" or open an
agent on a dashboard and see its live status. Two concrete uncertainties gate
everything else, and this spike exists to reduce them:

1. **What is an "agent instance" mechanically?** A record the server drives via
   the existing codex-per-turn machinery, or a detached long-lived subprocess
   (a `claude`/`codex` process in a sesh/tmux pane) that scufris supervises? The
   caller asked the spike to decide.
2. **How does the orchestrator observe a running agent?** v1 is explicitly
   **read-only** (observe + notify, not steer). What is the status source and
   the contract?

A good answer names the v1 agent runtime concretely, names the read-only status
mechanism concretely, says what happens to the just-built projects P0, and
leaves a phased task list that a `/plan` run can expand without re-litigating
the runtime choice. This is a **full re-baseline** of the projects-orchestrator
vision in `tasks/20260720-184150/SPIKE.md` (Revision 1) - that doc is superseded
by this one.

## Context

Grounded in a read of the current backend (`scufris/agent.py`, `sessions.py`,
`app.py`, `projects.py`) and frontend (`web/src/*-view.ts`,
`web/webpack.config.js`). The load-bearing facts:

- **scufris is a singleton today.** There is exactly one logical agent: a single
  `AgentHandle` (`agent.py:973`) holding a single `_session_id` pointer, and a
  single global `chat_lock = asyncio.Lock()` (`app.py:303`) that serializes
  every turn. Session listing hard-filters `originator in {codex_exec, scufris}`
  **and** `cwd == os.getcwd()` (`sessions.py:255-259`). The server's own cwd is
  the one and only project. **This singleton/one-cwd assumption is the thing
  multi-agent breaks, and generalizing it is the gating refactor.**
- **Subprocess-per-turn, no daemon.** Each turn spawns `codex exec --json`
  (turn-level) or `codex app-server` (JSON-RPC, token streaming), runs to
  completion, and exits (`agent.py:553-669`). There is no long-running agent
  process between turns. A turn is agentic on its own: `codex exec` uses tools
  until the task is done, so **one turn can already be a whole autonomous unit
  of work** (e.g. a prompt that says "use the flow skill to implement X").
- **Codex owns session state; scufris reads it.** Codex writes rollout JSONL to
  `$CODEX_HOME/sessions/rollout-*-<session_id>.jsonl`. scufris already parses
  these read-only for the transcript (`sessions.py:403`), a context snapshot
  (turns / tool calls / input+output tokens, `sessions.py:121`), usage/quota,
  and a memory footprint. **Per-agent read-only status is therefore ~80% already
  built** - each agent needs its own `session_id`, and status is a tail of that
  agent's rollout.
- **Backends are already abstracted.** `SCUFRIS_AGENT_BACKEND` selects
  `app_server` | `exec` | `mock` behind the `Agent` protocol (`agent.py:99`,
  `945`). MCP servers are registered per-invocation via `-c` flags with
  approval mode `never` (unattended). Turns run `--sandbox read-only` (set on
  turn 1, inherited on resume, `agent.py:436`) - **agents cannot write files
  today.** `agent_timeout_seconds` defaults to 120s and kills the subprocess
  (`agent.py:502`) - **too short for a long autonomous run.**
- **The frontend has 4 pages**: Agent (the real product - a streaming chat with
  sessions + fork), Projects (P0: CRUD + per-project tatr tasks), Stats (host
  metrics, polled 2s with client-side sparkline history), Settings (agent
  config console). Established patterns: pure `renderX(root, data, actions)` +
  injected actions, `sendJson` then reload, SSE for the chat stream, polling for
  stats. **Agent is chat-only (no orchestration breadcrumbs) and Stats is
  host-only (no per-agent view) - that disconnect is why they feel like
  gimmicks.**
- **Projects P0 (just landed).** `projects.py` persists `{id, cwd, name,
  language, description}` with CRUD and a per-project tatr-tasks endpoint, plus a
  `/projects/` page. Under the re-baseline this record is exactly the right "what
  repo does an agent work in" data, but Project stops being the destination and
  becomes **plumbing behind agent creation** (a project picker).

## Options considered

### Q1 - What is an agent instance?

- **A. In-process record + codex-per-turn driver (server-native).** An agent is
  a persisted record `{id, name, project_cwd, backend, model, goal|task_id,
  session_id, state}`; the server runs its turns with the *existing*
  subprocess-per-turn machinery, keyed by agent id instead of the singleton. An
  autonomous "work a goal" run is one long `codex exec` turn whose prompt invokes
  `/flow`. Status = tail that agent's rollout.
  - Pros: reuses ~everything (backend abstraction, streaming, rollout parsing);
    read-only status falls out for free; smallest new surface. cwd is handled by
    passing the project cwd to the subprocess (codex supports `-C`/cwd; today it
    just inherits `os.getcwd()`).
  - Cons: the server process must stay up while agents run (already true); a
    "running" agent is a live subprocess only for the duration of its turn -
    fine for a single long autonomous turn, but a multi-turn driver loop is
    future work; writing files needs the read-only sandbox lifted (a real
    decision, see risks).
  - Unknowns: does one long `codex exec` turn running `/flow` behave well
    unattended for many minutes (tool approvals, timeout, memory)? Needs a probe.

- **B. Detached supervised subprocess (sesh-style, backend-heterogeneous).** An
  agent is a long-lived OS process scufris launches and supervises - e.g.
  `claude -p` headless or a codex process in a tmux/sesh pane - running
  autonomously; scufris tracks pid/alive/dead and observes by tailing the
  process's output (or its rollout).
  - Pros: the only model that natively runs heterogeneous backends (Claude Code
    *and* codex) since each is just a subprocess; survives a scufris restart;
    matches the user's sesh mental model directly.
  - Cons: heaviest new infra - process lifecycle (spawn, monitor, kill, restart,
    orphan reaping), and observation is per-backend (Claude Code's
    `stream-json` vs codex rollout JSONL are different formats). This is where
    the real research/risk lives.
  - Unknowns: the write-access story, the resource/cost ceiling with N live
    processes, per-backend status normalization.

- **C. Hybrid: commit to the record + status contract now, make the runner
  pluggable.** The agent is a first-class record with a lifecycle state machine
  (`idle|running|blocked|done|error`) and a uniform read-only **status
  contract**; *how* it runs is a pluggable runner. v1 runner = A (in-process
  codex driver, reuses everything). A later runner = B (detached, Claude Code)
  slots behind the same record + contract once the loop is proven.
  - Pros: gets the dashboard, the orchestrator observation, and the end-to-end
    "goal -> agent -> /flow" loop working on the cheap A runner, while the
    abstraction that makes B possible is in place from day one; defers the
    genuinely hard/risky parts (heterogeneous backends, steering, per-agent
    resource model) behind a boundary instead of a rewrite.
  - Cons: the status contract has to be designed to fit both rollout-tailing and
    stream-json, which is a bit more upfront thought than hard-coding to codex.

### Q2 - How does the orchestrator observe an agent (read-only v1)?

- **Rollout tail (reuse existing parsing).** Each agent has a `session_id`;
  status = read the tail of its rollout for last activity, current tool call,
  turn/token counts, and a terminal marker. Reuses `sessions.py` wholesale.
  Pro: near-free, already proven. Con: codex-specific (fine for the A runner;
  the status contract abstracts it for B later).
- **Agent-written status file.** Each agent writes a small `status.json`
  (phase, current step, progress) that scufris reads. Pro: backend-agnostic,
  richer than rollout inference. Con: requires the agent to cooperate (a skill
  or tool that writes it); more moving parts than needed for v1.
- **Live IPC / event stream.** Agents push structured events to scufris over a
  socket/queue. Pro: real-time, needed eventually for notifications. Con: real
  infrastructure; overkill for read-only v1 status.

### Do nothing

Keep P0's Project page and the singleton agent. Cost: the AGENT and STATS pages
stay disconnected gimmicks, and the actual use case the user has (a cockpit for
parallel agent work) never gets built. Rejected - the user has explicitly
re-baselined toward the orchestrator.

## Decisions locked (revision 1, 20260720)

After the spike's first draft, the operator refined the direction and locked
four choices that this revision folds in. They tighten, not replace, the
option-C recommendation below:

1. **Multi-backend is a v1 design constraint, not a deferred phase.** There is a
   common `AgentBackend` interface (`run` / `stream` / `status` / `resume`) and
   the "main part" (orchestration, store, dashboard, status) is agnostic to
   which backend a given agent uses. Build the **codex** runner first (reuses
   everything), then a **claude** (Claude Code headless) runner *right after* -
   the second backend is what proves the interface is not accidentally
   codex-shaped. The interface must hide the four places the two differ: output
   format (codex rollout JSONL vs `claude -p --output-format stream-json`),
   session resume (codex thread-id vs `claude --resume`, different on-disk
   stores), MCP config (codex `-c` vs `claude --mcp-config`), and the
   permission/sandbox model (codex `--sandbox` vs claude
   `--permission-mode`/`--allowedTools`).
2. **No request-scoped timeout. Runs are background jobs; the frontend polls.**
   The 120s kill is an artifact of running the turn *inside the held HTTP
   request under the global lock*. v1 decouples the run from the request: an
   endpoint enqueues a job, an **in-process supervisor** runs the agent
   subprocess in the background with a **concurrency cap acting as the queue**
   (agents past the cap wait), and status is **polled** (the rollout tail is
   cheap). No external broker (no Redis/Celery). The hard wall-clock timeout is
   replaced by a **per-agent budget + a liveness/heartbeat** that only catches a
   genuinely stuck subprocess. This splits interaction into two modes:
   interactive orchestrator turns stay SSE-streamed and short; autonomous agent
   runs are background + poll.
3. **Agents may write to their project, gated per-agent.** "Implement feature X
   in project Y" requires writing files, so v1 agents can lift the read-only
   sandbox - but write is an explicit **per-agent opt-in scoped to the project
   cwd**, surfaced in the UI, not a silent default. (This is a *different*
   read-only than decision-by-the-operator "read-only orchestrator observation",
   which still holds: the orchestrator observes agents, it does not steer them.)
4. **The orchestrator is itself backend-swappable.** The main chat agent routes
   through the same `AgentBackend` interface, so codex or claude can drive the
   orchestrator too; it is not a special case outside the interface.

## ADR-001: execution and delivery - workers + per-agent event bus + SSE

- STATUS: ACCEPTED (revision 1)

### Context

Decision 2 removes the request-scoped timeout by running agents as background
jobs. That raised a fair question: if agents run in workers and the dashboard
polls, do we lose SSE streaming from the agent to the browser? The worry is that
"multiple concurrent agents" forces a worker model, and a worker model forces
polling, so live streaming is lost.

The premise is half right. **Concurrency does need workers** (background
execution decoupled from any single request, so one long run neither blocks
others nor dies when its request ends). But **execution and delivery are
orthogonal axes**, not one axis:

- Execution: is the agent subprocess tied to an HTTP request (today) or to a
  supervisor that outlives requests (workers)?
- Delivery: do browser updates arrive by polling `GET` or by a held SSE stream?

You can pick workers on the first axis AND SSE on the second. The trick is to
stop letting the SSE request *be* the thing that runs the agent. Split them: one
request STARTS the run (returns immediately), a separate SSE request SUBSCRIBES
to the run's events. Between producer (worker) and consumer (SSE) sits a
per-agent **event bus**.

### Decision

The worker publishes every normalized `StreamEvent` to a per-agent in-memory
event bus (a fan-out ring buffer). The SSE endpoint is a thin **subscriber** that
replays the recent buffer on connect (via `Last-Event-ID`) and then forwards
live events. Because the SSE request no longer runs the agent, it can drop and
reconnect freely without touching the run, and N viewers can watch the same
agent. The backend's own durable log (codex rollout / claude session jsonl) is
the restart-safe replay source behind the in-memory buffer. We use **both**
delivery modes where each fits: the dashboard LIST polls coarse status (cheap, no
N open streams); the FOCUSED agent view opens one SSE stream for live tokens.

### Flow

```
                              BROWSER
    dashboard list (many agents)          agent detail (one, focused)
    GET /api/agents  --poll ~2s-->        GET /api/agents/{id}/events  (SSE)
         |                                        ^   live events + replay
         |                                        |   (Last-Event-ID, drop-safe)
         v                                        |
 +===================================  FastAPI  ==+============================+
 |  POST /api/agents/{id}/run   -> enqueue, return run id immediately         |
 |  GET  /api/agents            -> read status contract (poll)                |
 |  GET  /api/agents/{id}/events-> SUBSCRIBE to that agent's EventBus (relay) |
 +==================================|=========================================+
                                    | start job (does NOT hold the request)
                                    v
                          +---------------------+   concurrency cap = semaphore
                          |     SUPERVISOR      |-----------------------------+
                          +---------------------+                             |
                             |          |          |                         |
                             v          v          v                         |
                       AgentWorker  AgentWorker  AgentWorker  (bg tasks)      |
                             |          |          |                         |
                    AgentBackend.run/stream  (codex | claude subprocess)     |
                       publishes normalized StreamEvents                     |
                             |          |          |                         |
                             v          v          v                         |
                    per-agent EventBus (in-mem fan-out ring buffer) <--------+
                             |  also appended durably (rollout / session jsonl)
                             v
                    subscribers: any open SSE relay(s); replay on reconnect;
                    restart -> rebuild buffer tail from the durable log
```

### Consequences

- Timeout gone (execution is not request-bound); SSE kept (delivery layered on
  the bus). Both of the operator's wants satisfied at once.
- Multiple agents run concurrently under the semaphore; multiple viewers per
  agent; a dropped browser never kills a run.
- The bus takes normalized `StreamEvent`s, so it is backend-agnostic - codex and
  claude both publish to the same bus (fits the A2 interface).
- Slightly more moving parts than pure polling: a per-agent buffer + subscribe
  lifecycle. Mitigated by keeping the buffer small and using the durable log for
  anything older than the buffer window.

### Alternatives rejected

- **Workers + poll only (no SSE).** Simpler, but focused live token streaming
  becomes choppy at a 1-2s poll cadence; the operator explicitly wants to keep
  SSE.
- **Keep request-scoped SSE, just raise the timeout.** One held connection per
  run, still under a lock; a dropped connection risks the run; does not scale to
  many concurrent agents. This is the model decision 2 moves away from.
- **External broker (Redis/Kafka) as the bus.** Real pub/sub, restart-durable,
  but heavyweight for a single-operator local tool; the in-memory bus + on-disk
  log gives the same properties at this scale.

## Recommendation

**Adopt option C with the A runner for v1 and rollout-tail status**, as refined
by the locked decisions above (multi-backend interface from day one, background
supervisor + poll instead of a request timeout, gated per-agent write) and
ADR-001 (workers + event bus + SSE, so streaming is kept). The original
recommendation text stands; concretely:

1. **Agent becomes the first-class entity; Project becomes plumbing.** Add an
   `AgentStore` (`agents.json`, mirroring `projects.py`/`settings_store.py`)
   persisting `{id, name, project_cwd, backend, model, goal|task_id, session_id,
   state}`. Project records stay as the picker behind "which repo"; the
   standalone `/projects/` page folds into the agent-creation flow rather than
   living as its own destination. The tatr-tasks endpoint stays useful (bind an
   agent to a task).

2. **A uniform read-only status contract, backed by rollout-tail for v1.** A
   `agent_status(agent_id) -> {state, last_activity, current_tool, turns,
   tokens, updated_at}` computed from the agent's rollout (reuse `sessions.py`).
   The contract is the abstraction seam so a detached/Claude-Code runner can
   fill it differently later.

3. **The end-to-end proof: goal -> agent -> `/flow`.** Creating an agent with a
   goal kicks off one autonomous `codex exec` turn whose prompt invokes the flow
   skill, scoped to the project cwd, tracked by state + rollout. This is the
   smallest thing that demonstrates the whole vision and forces the two real
   blockers into the open (below).

4. **The orchestrator observes via new MCP tools.** Give the main agent (the
   existing chat page) read-only tools `list_agents` and `agent_status(id)` so
   "what is agent-2 doing" is answered by reading its status. No steering in v1.

5. **A dashboard/Agents page** lists agents with live status (state, last
   activity, tokens), reusing the Stats polling + sparkline patterns. STATS and
   AGENT stop being disconnected once agents are the thing they are about.

Why C over A-only: A-only hard-wires to codex and would need a rewrite to add
Claude Code; the record + status-contract boundary is cheap now and is the whole
point of the "multiple agents, multiple backends" vision. Why C over B-first: B
is the expensive, risky half (process supervision, per-backend status, resource
model); doing it first means building infrastructure before proving the loop is
even useful. C proves the loop on the cheap runner, then B slots in behind the
boundary as a deliberate later phase.

**Two blockers this direction must confront honestly (they are real, not
hand-waved):**

- **The singleton/one-cwd assumption.** `cwd == os.getcwd()` session filtering
  and the global `chat_lock` must generalize to per-agent cwd and per-agent
  locking before two agents can run on two projects. This is the gating refactor
  and is the first real task (A0 below).
- **Read-only sandbox + short timeout.** Agents run `--sandbox read-only` and
  die at 120s. Both are lifted in v1 per the locked decisions: the timeout goes
  away with the background-supervisor model (decision 2), and the sandbox is
  lifted **per agent via an explicit, cwd-scoped write opt-in** (decision 3) so
  "vibe coding" works while write stays a visible, gated choice rather than a
  silent default.

## Open questions

- **Does one long `codex exec` turn running `/flow` behave unattended?** Tool
  approvals are `never` (good) but a multi-minute agentic turn's timeout, memory
  growth, and failure modes are unverified. Resolve with a probe (A2) before
  committing the UI to it.
- **One server, many cwds, vs one server per project.** Generalizing the cwd
  filter (A0) keeps a single server but touches session code broadly; running
  one scufris per project sidesteps it but multiplies deployment. Lean
  single-server + per-agent cwd, but confirm during A0.
- **Concurrency ceiling / cost.** N concurrent live turns each burn tokens and
  hold a subprocess. Need a max-concurrent cap and a per-agent cost readout.
  Not a v1 blocker but must be named in the dashboard.
- **Write access and the sandbox** - deferred to its own phase behind an
  explicit gate; the "vibe coding" dream needs it but it is the biggest safety
  surface.
- **Heterogeneous backends (Claude Code) and steering** - explicitly deferred
  to the B runner behind the status contract; likely its own follow-up spike.

## Next steps

Direction-level tasks this spike seeded, for `/plan` to break into steps. Phased
so each phase is independently landable and the risky parts are isolated.

- tatr 20260720-221922 (A0, foundation): agent runtime foundation -
  de-singleton (per-agent cwd instead of `cwd == os.getcwd()`) AND background
  execution: turn the global `chat_lock` into an in-process supervisor with a
  concurrency cap, decouple runs from the HTTP request, poll for status, and
  replace the 120s request timeout with a per-agent budget + liveness/heartbeat
  (decisions 1-2). Gating refactor; kills the timeout.
- tatr 20260720-221929 (A1): `AgentStore` (`agents.json`) - agent as a
  first-class record `{id, name, project_cwd, backend, model, goal|task_id,
  session_id, state, write_enabled}` with CRUD, mirroring `projects.py`; Project
  demoted to the project-picker plumbing behind agent creation. [dep: A0]
- tatr 20260720-221935 (A2, interface + codex + probe): the common
  `AgentBackend` interface (`run`/`stream`/`status`/`resume`), the **codex**
  runner behind it, read-only `agent_status` via rollout-tail, and the probe
  that runs one long autonomous `codex exec` turn invoking `/flow` on a scratch
  project to verify unattended behaviour (approvals, memory, liveness). The
  orchestrator (main chat) also routes through this interface (decision 4).
  [dep: A1]
- tatr 20260720-223938 (A2b, claude runner): a **claude** (Claude Code headless,
  `claude -p --output-format stream-json`) runner behind the same
  `AgentBackend` interface - session resume, MCP config, permission model, and
  its status source (stream-json / session jsonl) normalized to the contract.
  Building the second backend proves the interface is not codex-shaped
  (decision 1). Own research + probe. [dep: A2]
- tatr 20260720-221942 (A3): create-agent-with-goal end to end - bind an agent
  to a project + goal (with the per-agent, cwd-scoped **write** opt-in,
  decision 3), launch it as a background job via the supervisor, track state via
  the status contract. The vision's first real vertical slice. [dep: A1, A2]
- tatr 20260720-221951 (A4): the Agents dashboard page - list agents with live
  status (state, last activity, tokens) via polling, reusing the Stats
  polling/sparkline patterns; fold the standalone Projects page into the
  agent-creation flow. [dep: A1, A3]
- tatr 20260720-221957 (A5): orchestrator observation - read-only `list_agents`
  / `agent_status` MCP tools so the main chat agent can answer "what is agent-N
  working on". [dep: A2]

Deferred behind their own later phases/spikes (named so they are not forgotten,
NOT seeded as ready work): steering (bidirectional control of a running agent -
the operator locked read-only observation for v1); notifications / push event
stream (v1 polls); per-agent cost accounting and richer concurrency policy
beyond a flat cap. (Multi-backend and gated write-access were deferred in the
first draft but are pulled into v1 by decisions 1 and 3.)

## Fix record

(Filled by each implementing task as it lands.)
