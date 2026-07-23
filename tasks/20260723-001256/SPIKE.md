# Spike: bidirectional agent<->orchestrator communication (async, needs-input, wake)

- DATE: 20260723-001256
- STATUS: RECOMMENDED
- TAGS: spike, agents, backend, mcp

## Question

The orchestrator drives sub-agents, but communication is one-way and synchronous:
`message_agent` (POST `/api/agents/{id}/chat`) blocks up to 120s collecting the
reply SSE. There is no way for a sub-agent to (a) run in the background while the
orchestrator does other things, or (b) INITIATE contact - ask a question, signal
"I'm blocked, approve the merge?". The orchestrator cannot poll for "which agents
need me", and nothing wakes it when one does.

Decide, concretely enough that `/plan` can expand without re-litigating:

1. The transport/primitive - replace or augment `message_agent`?
2. The sub-agent signaling mechanism - a scoped callback tool, or infer the
   state from how the turn ends?
3. The needs-input state + the question payload shape.
4. The orchestrator wake mechanism - poll, or a dashboard bridge that grants a
   turn.

## The acceptance scenario (the exact case the design must make work)

A hello-world sub-agent stops before merging to master, waiting for confirmation.
The orchestrator could not tell "blocked, needs input" from "done" and had no way
to poll, so the loop stalled. The chosen design must make this exact case work,
end to end, WITHOUT a human babysitting the blocking `message_agent` call.

## Context (grounded in the code)

- **`run_agent` is already fire-and-forget.** `POST /api/agents/{id}/run`
  (`app.py:1161-1183`) calls `_launch_agent_turn` and returns the queued status
  immediately (the bus is discarded). The async LAUNCH half exists; the
  OBSERVE-LATER and SIGNAL-BACK halves do not.
- **`message_agent` blocks.** The MCP tool (`mcp_server.py:477-515`) POSTs to
  `/api/agents/{id}/chat` with `_CHAT_TIMEOUT = 120.0` and drains the SSE until
  the final reply. `agent_chat` (`app.py:1244-1258`) launches the turn and
  relays the per-run bus via `_relay_bus_sse` (`app.py:1208-1226`).
- **The EventBus is PER-RUN and EPHEMERAL.** `supervisor.start` creates one bus
  per run (`supervisor.py:181-219`), `_execute` calls `bus.close()` on completion
  (`supervisor.py:250-293`), and `GET /api/agents/{id}/events`
  (`app.py:1228-1242`) returns 404 when no run is active. Event kinds are live
  stream deltas only - `text_delta`, `reasoning_delta`, `tool`, `done(reply,
  session_id)`, `error` (`agent.py:64-93`). CONSEQUENCE: a "needs input" signal
  that must outlive the run (the agent ended its turn to WAIT) cannot live on the
  bus - the bus is gone by the time the orchestrator would read it. The signal
  must be DURABLE.
- **Run completion is already captured, durably, by `mark_finished`.** The
  on-complete callback (`app.py:1112-1132`) calls
  `agents.mark_finished(id, state=, session_id=, backend=)`
  (`agent_store.py:467-504`), which persists the terminal state and writes the
  session to the `SessionRegistry` (`sessions.json`,
  `agent_store.py:103-167`, the just-landed 20260723-001251). This is the seam
  where a durable OUTCOME (final message + state) can be recorded too.
- **After T1/T3 sub-agents have NO scufris MCP tools.** `_mcp_overrides`
  (`agent.py:153-194`) registers the scufris server ONLY when `is_orchestrator`
  (`app.py:1106`, `agent.id == ORCHESTRATOR_ID`). Regular agents draw tools from
  their own project `.config`/`.skills`. Adding ANY sub-agent-facing scufris tool
  means widening that gate - but T3 was a CAPABILITY preference (no
  create/run/control for sub-agents), not a hard "sub-agents get nothing"
  boundary, so a role/audience allowlist that exposes only a notify tool honours
  T3's real guarantee (see BC2 `DECISION.md`).
- **The MCP server is a separate subprocess** and cannot touch the live
  in-process `Supervisor`; the existing control tools (T2) reach back into the
  app over the local HTTP API (`127.0.0.1:<port>`). Any new tool - sub-agent or
  orchestrator - obeys the same constraint.
- **The orchestrator is turn-based and unpushable.** `_launch_agent_turn`
  (`app.py:1074-1145`) is the ONLY way to grant a turn; it raises **409 if a run
  for that agent is already queued/running** (`app.py:1088-1093`) and reserves
  `serialize_key=agent.id` (`supervisor.py:145-179`). Nothing can inject an
  unsolicited message into a running turn. Session mutations
  (`new`/`switch`/`delete`/`reset`) hold `supervisor.serialized(ORCHESTRATOR_ID)`
  (`app.py:1483,1543,1665`), and the code already notes that wrapping a LAUNCH
  inside that same key self-deadlocks (`app.py:1509-1510`; lesson
  `nested-serialized-key-self-deadlock`). CONSEQUENCE for any waker: to act on a
  signal it must GRANT the orchestrator a turn via `_launch_agent_turn`, must NOT
  already hold `ORCHESTRATOR_ID`, and must cope with the 409 (the orchestrator
  may be mid-turn) by deferring the wake, not dropping it.
- **Models stop in PROSE when blocked.** The acceptance scenario is a turn that
  ends `DONE` with a final message like "should I merge? waiting for
  confirmation" - there is no tool call to intercept. Any design that depends on
  the sub-agent RELIABLY calling a "I'm blocked" tool is betting against how the
  models actually behave.

## The problem, decomposed

1. **Async / non-blocking:** `run_agent` already launches without waiting; add
   the OBSERVE-LATER half.
2. **Agent -> orchestrator signaling:** no channel today, and sub-agents have no
   scufris MCP tools (T3). Either re-introduce a narrow sub-agent callback tool
   or INFER the state from how the turn ends.
3. **A real needs-input state:** `AgentState.BLOCKED` exists but means "waiting on
   an APPROVAL" (`enums.py:52`), not "ended a turn awaiting a decision".
   `agent_status` exposes only `last_message`, so done vs waiting-for-me is
   indistinguishable.
4. **Waking the orchestrator (the deep one):** grant it a turn when a sub-agent
   needs input - a bridge (a signal enqueues an orchestrator turn with the
   question injected) or a poll the orchestrator runs itself.

## Options considered

### Q2 - Sub-agent signaling: explicit callback tool vs inferred completion

GATE DECISION (2026-07-23): the user chose the **explicit callback tool** over
inference. Both are recorded below; the explicit tool is the CHOSEN v1
mechanism, with inference kept as the always-on completion SUBSTRATE (the
durable outcome record captures every turn's end regardless, so a turn that ends
without calling the tool still leaves a `DONE` outcome + final message the
orchestrator can inspect). The two compose: the tool hard-sets `WAITING` with a
structured payload; the substrate catches everything else.

- **Explicit narrow callback tool (CHOSEN)** (`request_input(question)` /
  `notify`), the ONE scufris tool sub-agents get, HTTP-backed like the T2 control
  tools. Gives a hard, structured signal. Delivery mechanism decided in BC2's
  `DECISION.md` (role-scoped, Option B): ONE scufris server, the `is_orchestrator`
  gate generalized into a role/audience (`orchestrator` vs `agent`) with tools
  tagged by audience, NOT a second server. This reframes T3
  (`tasks/20260722-222729`) as what it actually was - a CAPABILITY preference
  ("none of the current tools are useful for sub-agents; don't let them
  create/run agents"), not a hard "sub-agents get nothing" security boundary -
  so exposing one `agent`-audience tool via an explicit allowlist preserves T3's
  real guarantee (no control tools for sub-agents) without a physical boundary.
  Costs: (a) claude sub-agents get no scufris MCP at all today (`backends.py`
  never adds it), so the tool is codex-first - claude parity needs a
  `--mcp-config` wiring, tracked as a follow-up in the seeded task; (b) it bets
  the model calls it when blocked, so the inference substrate below stays on as
  the backstop for turns that just stop in prose.
- **Infer from turn completion (substrate, not the primary signal).** Every
  sub-agent turn ends with a durable `mark_finished(state, final message)`. The
  durable outcome record (BC1) captures this for ALL turns, so the orchestrator
  can always read a finished agent's final message and classify it even when the
  tool was not called. Kept as the backstop under the explicit tool rather than
  the primary mechanism.

### Q1 - Transport / primitive: replace or augment message_agent

- **Replace `message_agent` with a job/inbox abstraction** - large, and throws
  away a primitive (blocking synchronous Q&A) that is still the right tool when
  the orchestrator WANTS to wait for a short reply.
- **Augment (recommended).** Keep `run_agent` (async launch) and `message_agent`
  (synchronous steer) as-is. Add ONE durable layer - a per-agent run OUTCOME
  persisted at `mark_finished` - and TWO orchestrator-only reads over it:
  `pending_agents()` (who has an unacknowledged needs-input/error outcome, with
  the message) and an `acknowledge(agent_id)` to clear it. The "job with an
  outbox" the candidate shape wanted, minus a new bus: the outbox is the durable
  outcome record, not the ephemeral EventBus.

### Q3 - needs-input state + payload shape

- **Overload `BLOCKED`** - it already means approval-gating; overloading it
  muddies the one place that word is used.
- **Add `AgentState.WAITING` (recommended)** = "ended a turn awaiting a decision",
  distinct from `DONE`. Set at `mark_finished` time. The payload is a durable
  per-agent OUTCOME record (a sidecar `outcomes.json` owned by `AgentStore`,
  mirroring the `SessionRegistry` sidecar pattern - atomic write, tolerant load):
  `{ agent_id, run_id, session_id, state, message, ts, acknowledged }`. Who sets
  `WAITING` vs `DONE`: baseline v1 records the raw terminal state and the final
  message and lets the ORCHESTRATOR classify on read (so `WAITING` may initially
  be derived by the orchestrator/`pending_agents` heuristic from a non-empty
  trailing question, not hard-set). The enum value is added now so the state has
  a name the UI and the future explicit tool can both target.

### Q4 - Wake mechanism: poll vs bridge

- **Poll only.** The orchestrator calls `pending_agents()` at the end of each of
  its own turns. Simple, no new app machinery - but it only fires while SOMETHING
  is already driving the orchestrator (a human, or Telegram T4). It does NOT wake
  a truly idle orchestrator, so the acceptance scenario (loop stalled, nobody
  driving) still stalls. Good as the DEGRADED mode and the v1 first step.
- **Dashboard bridge (recommended for full acceptance).** An in-app async watcher
  observes run completions (it can hook the same on-complete seam, or drain the
  outcomes store) and, on a needs-input/error outcome, GRANTS the orchestrator a
  turn via `_launch_agent_turn(orchestrator, injected_prompt)` where the prompt
  carries the sub-agent id + its question. Constraints from the map: the watcher
  must NOT hold `ORCHESTRATOR_ID` (self-deadlock), must handle the 409 (defer the
  wake into its own small pending-wake queue and retry when the orchestrator goes
  idle, so a wake is never dropped), and should BATCH (multiple completions while
  the orchestrator is busy fold into one "these agents need you: [...]" turn).
  Config-gated (`auto_wake` on/off); when off, the orchestrator falls back to
  polling. This is what makes the idle loop self-heal.

## Recommendation (the decision)

Build the "run as a job with a durable outcome + an explicit sub-agent callback
signal + a wake bridge" design (as decided at the gate):

- **Explicit sub-agent callback (the gate's choice).** Give sub-agents ONE
  narrow, notify-only scufris tool `request_input(question)` that HTTP-POSTs a
  needs-input signal, hard-setting the agent's `WAITING` outcome with a
  structured question payload. Delivered via a ROLE-SCOPED tool model (BC2
  `DECISION.md`, Option B): one scufris server, the `is_orchestrator` gate
  generalized into an `orchestrator`/`agent` audience with tools tagged by role -
  not a second server. T3 is reframed as a capability preference, so this adds an
  `agent`-audience tool via an explicit allowlist rather than reversing a
  security boundary. Codex-first, with claude MCP wiring as a tracked follow-up.
  Run COMPLETION (the durable outcome record below) stays on as the backstop for
  turns that stop in prose without calling the tool.
- **Augment, don't replace.** Keep `run_agent` and `message_agent`. Add a durable
  per-agent OUTCOME record written at `mark_finished` (final message + terminal
  state + run/session id + `acknowledged` flag), a sidecar `outcomes.json` owned
  by `AgentStore` mirroring the `SessionRegistry` pattern. This is the substrate
  the callback tool sets and the orchestrator reads.
- **Name the state.** Add `AgentState.WAITING` = "ended a turn awaiting a
  decision". Surface it (and the final message) via an orchestrator-only
  `pending_agents()` MCP tool (HTTP-backed, T2 style) plus `acknowledge(agent_id)`
  to clear.
- **Wake via a config-gated dashboard bridge** that grants the orchestrator a
  turn on a needs-input/error outcome, injecting the question; polling
  (`pending_agents()` at end-of-turn) is the always-available fallback. The
  bridge owns a pending-wake queue to absorb the 409 and batch concurrent
  completions, and never holds `ORCHESTRATOR_ID` when it launches.

Why this shape: it makes the acceptance scenario work with a hard, structured
"I'm blocked" signal, while keeping the rest of the architecture intact (the
SessionRegistry sidecar pattern, the T2 HTTP-backed-tool convention, the
serialize-key discipline). The one deliberate change - exposing a single
`agent`-audience scufris tool via a role allowlist - is scoped to a
capability-free notify callback, not the control/observe surface, honouring T3's
real guarantee (BC2 `DECISION.md`).

Land this BEFORE the Telegram bot (T4/T5): "the agent got stuck and no one knew"
surfaces immediately over chat, and the durable outcome + `pending_agents` are
exactly what a Telegram-driven orchestrator polls.

## Open questions handed to /plan (not blockers)

- Where the bridge watcher hooks: reuse the `on_complete` callback seam
  (`app.py:1112-1132`) directly, or have it drain the outcomes store on a tick?
  The callback seam is lower-latency and already carries the launch-time snapshot;
  the store-drain is more decoupled and survives a missed callback. Lean: hook the
  seam, persist first so a drain can recover.
- `WAITING` is hard-set by the `request_input` callback (BC2), not inferred at
  `mark_finished`. The substrate outcome still records the raw terminal state +
  final message for turns that end WITHOUT calling the tool, which the
  orchestrator can classify on read. Open sub-question for /plan: does
  `request_input` return immediately (fire-and-forget signal, the sub-agent turn
  then ends) or block awaiting the orchestrator's answer? Lean: return
  immediately and let the sub-agent end its turn, then resume it with the answer
  - a blocking `request_input` re-creates the 120s-wait problem inside the
  sub-agent.
- Auto-wake default: off (opt-in) for v1, given the orchestrator now defaults to
  `auto` permission mode (20260723-001243) and a wake grants it an unattended
  turn. Confirm at plan/gate.

## Seeded tasks (direction-level; /plan breaks each into steps + DoD)

The flow's `/plan` phase expands these into Steps + DoD; priorities slot under
this spike (p40), in dependency order. Seeded as tatr tasks:

| task | id | pri |
|------|----|-----|
| BC1 durable outcome + `WAITING` | `20260723-094258` | 39 |
| BC2 `request_input` callback tool | `20260723-094303` | 38 |
| BC3 `pending_agents`/`acknowledge` | `20260723-094308` | 37 |
| BC4 wake bridge | `20260723-094313` | 36 |
| BC5 e2e example + acceptance test | `20260723-094318` | 35 |

- **BC1 - Durable run-outcome record + `AgentState.WAITING` (backend).** At
  `mark_finished`, persist a per-agent outcome (`state`, final `message`,
  `run_id`, `session_id`, `ts`, `acknowledged=false`) in an `outcomes.json`
  sidecar owned by `AgentStore` (atomic write + tolerant load, mirroring
  `SessionRegistry`). Add `AgentState.WAITING`. This is the substrate the rest
  build on. Test: after a fake sub-agent run ends, the outcome (message + state)
  is readable and survives a simulated restart. (depends on: none)
- **BC2 - `request_input` sub-agent callback tool (the chosen signal).** Expose
  ONE notify-only tool, `request_input(question)`, to sub-agents via a ROLE-SCOPED
  model (BC2 `DECISION.md`, Option B): generalize the `is_orchestrator` gate in
  `_mcp_overrides` (`agent.py:153-194`) into an `orchestrator`/`agent` audience,
  tag tools by role, register the one scufris server for sub-agents with only the
  `agent`-audience tool. It HTTP-POSTs a needs-input signal to the app (T2
  pattern), hard-setting the caller's `WAITING` outcome with a structured
  question payload. Codex-first; note+track the claude `--mcp-config` parity gap.
  Test: a sub-agent calling `request_input` leaves a `WAITING` outcome with the
  question; the `agent` role exposes ONLY `request_input` (no control tools).
  (depends on: BC1)
- **BC3 - `pending_agents()` + `acknowledge()` orchestrator-only MCP tools.**
  HTTP-backed (T2 style): `pending_agents()` returns agents with an
  unacknowledged needs-input/error outcome and their final message;
  `acknowledge(agent_id)` clears it. Register orchestrator-only. Test: a
  `WAITING` (or finished) sub-agent shows up in `pending_agents()`, disappears
  after `acknowledge`. (depends on: BC1)
- **BC4 - The wake bridge (backend).** A config-gated in-app watcher that, on a
  needs-input/error outcome, grants the orchestrator a turn via
  `_launch_agent_turn` with the sub-agent id + question injected; owns a
  pending-wake queue that absorbs the 409 and batches concurrent completions;
  never holds `ORCHESTRATOR_ID` at launch. Test (async httpx, two concurrent
  runs): a sub-agent `request_input` enqueues exactly one orchestrator turn
  carrying the question, even while the orchestrator is mid-turn. (depends on:
  BC1, BC2)
- **BC5 - End-to-end example + acceptance test.** An `examples/` script and an
  integration test that replay the acceptance scenario: launch a (faked)
  hello-world sub-agent that calls `request_input` awaiting merge confirmation
  -> the orchestrator is woken (bridge) / polls (`pending_agents`) -> answers by
  resuming the sub-agent's session -> the sub-agent proceeds. Proves the loop
  self-heals. (depends on: BC1-BC4)

## Notes

- Peer of the Telegram spike (`tasks/20260722-221359`); sequence BC1-BC4 before
  T4 (`tasks/20260722-222734`).
- Depends on the persisted `SessionRegistry` (`tasks/20260723-001251`, CLOSED) -
  stable session keys are what make "answer by resuming the sub-agent's session"
  reliable.
- Relevant lessons: `nested-serialized-key-self-deadlock` (the bridge must not
  hold `ORCHESTRATOR_ID`), `supervisor-endpoints-must-be-async`,
  `mark_finished-keys-by-launch-snapshot-backend`,
  `persist-callback-must-not-raise`.
</content>
</invoke>
