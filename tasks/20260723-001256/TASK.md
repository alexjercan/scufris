# Spike: bidirectional agent<->orchestrator communication (async, needs-input, wake mechanism)

- STATUS: CLOSED
- PRIORITY: 40
- TAGS: spike, agents, backend, mcp

## Story (spike)

The orchestrator drives sub-agents, but communication is one-way and synchronous:
`message_agent` (POST `/api/agents/{id}/chat`) blocks up to 120s collecting the reply
SSE. There is no way for a sub-agent to (a) run work in the background while the
orchestrator does other things, or (b) INITIATE contact - ask a question, send a
"questionnaire", or signal "I'm blocked, approve the merge?". The orchestrator can't
poll for "which agents need me", and nothing wakes it when one does. This spike
decides the mechanism for real back-and-forth, async, agent<->orchestrator
communication - i.e. "maybe send_message via MCP is not the right primitive; what is?".

## Concrete trigger (the acceptance scenario)

A hello-world sub-agent stopped before merging to master, waiting for confirmation.
The orchestrator could not tell "blocked, needs input" from "done" and had no way to
poll, so the loop stalled. The spike's chosen design must make this exact case work.

## The problem, decomposed (for the spike to solve)

1. Async / non-blocking: `run_agent` is already fire-and-forget; the missing half is
   observe-later, so the orchestrator isn't stuck holding a 120s call.
2. Agent -> orchestrator signaling: no channel today, AND after T3
   (`tasks/20260722-222729`) sub-agents have NO scufris MCP tools at all - so either
   re-introduce a NARROW sub-agent tool (`ask_orchestrator` / `notify`), or have the
   run-engine infer the state from how the turn ends.
3. A real "needs-input" state: `AgentState.BLOCKED` exists but is scoped to APPROVAL
   gating (`enums.py`), not "ended a turn awaiting a decision". `agent_status` only
   exposes `last_message`, so done vs waiting-for-me is indistinguishable.
4. Waking the orchestrator (the deep one): the orchestrator is a turn-based codex
   process; codex/MCP cannot push an unsolicited message into a running turn. Real
   back-and-forth needs SOMETHING to grant the orchestrator a turn when a sub-agent
   needs input - a dashboard BRIDGE (a needs_input event enqueues an orchestrator
   turn with the question injected) or a polling loop.

## A candidate shape to test (NOT a decision)

Model a sub-agent run as a JOB with an inbox/outbox on the existing EventBus, not a
chat peer: `run_agent` (async, exists) -> the job emits `progress` /
`needs_input(questions)` / `done` on its bus (the `/api/agents/{id}/events` SSE
already exists) -> a bridge wakes the orchestrator on `needs_input` -> the
orchestrator answers by resuming the session. Reuses the supervisor/EventBus and
turns "back and forth" into structured lifecycle events rather than free-form chat.

## Spike output

A `SPIKE.md` deciding: the transport/primitive (replace or augment `message_agent`),
the sub-agent signaling mechanism (scoped tool vs inferred state), the needs-input
state + question payload shape, and the orchestrator wake mechanism; plus seeded
implementation tasks. Recommend landing this BEFORE the Telegram bot (T4/T5) -
"the agent got stuck and no one knew" surfaces immediately over chat.

## Notes

- Peer of the Telegram spike (`tasks/20260722-221359`); sequence before T4
  (`tasks/20260722-222734`).
- Grounding: `message_agent` / `_relay_bus_sse` (SSE frames), the EventBus +
  supervisor (per-run bus), `/api/agents/{id}/events`, `AgentState` (`enums.py`),
  `agent_status` (`scufris/mcp_server.py`).
- Tension to resolve: T3 made the scufris MCP server orchestrator-only, so sub-agents
  currently have no callback tool - the signaling mechanism must account for that.
- spike-seeded direction; `/spike` first, then `/plan` the tasks it seeds.

## Spike output (close record, 2026-07-23)

`SPIKE.md` written (STATUS: RECOMMENDED). Decided all four questions:

1. Transport: AUGMENT, don't replace - keep `run_agent`/`message_agent`, add a
   durable per-agent OUTCOME record at `mark_finished` (the ephemeral per-run
   EventBus cannot hold a signal that outlives the run).
2. Signaling: GATE DECISION - an EXPLICIT narrow `request_input` sub-agent
   callback tool (chosen over inference), delivered via a role-scoped tool model
   (BC2 `DECISION.md`, Option B: one scufris server, `is_orchestrator` gate
   generalized into an `orchestrator`/`agent` audience - not a second server).
   T3 reframed as a capability preference, not a security boundary. The durable
   outcome stays on as the completion backstop.
3. State: add `AgentState.WAITING` ("ended a turn awaiting a decision"); hard-set
   by `request_input`; surfaced via orchestrator-only `pending_agents()` /
   `acknowledge()`.
4. Wake: a config-gated dashboard bridge that grants the orchestrator a turn via
   `_launch_agent_turn`, with a pending-wake queue to absorb the 409 and never
   holding `ORCHESTRATOR_ID`; polling is the fallback.

Seeded 5 tatr tasks (dependency order): BC1 `20260723-094258` (outcome +
`WAITING`), BC2 `20260723-094303` (`request_input`), BC3 `20260723-094308`
(`pending_agents`/`acknowledge`), BC4 `20260723-094313` (wake bridge), BC5
`20260723-094318` (e2e example + acceptance test). Land before Telegram T4/T5.

Per the user's gate (`/flow`, spike-only): stopped here. No app code shipped -
the diff is `SPIKE.md` + the 5 task files. The build (BC1-BC5, squash-merge each
to master) is a later `/flow` run over these tasks.
