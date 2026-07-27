# DECISION: how a backend StreamError surfaces through agent_status / pending_agents

- STATUS: ACCEPTED
- DATE: 2026-07-27

## Context

A backend yields a terminal `StreamError` event (idle timeout, over-limit line,
thread-setup failure). Today the supervisor's `_drain` only PUBLISHES that event
to the run bus; the async generator then completes normally, so `_execute` sets
`RunPhase.DONE` and leaves `run.error is None`. The single persist chokepoint
(`_launch_agent_turn.persist`) maps `DONE -> AgentState.DONE` with an empty
message (no `StreamDone` frame captured a reply). Net effect: a failed turn
persists as DONE with no diagnostic, invisible to `pending_agents` (which lists
only WAITING/REPORTED/ERROR) and to `agent_status` (which reads only the agent
record + backend `read_status`, never the outcome).

## The fork (task Step 1)

- Option A: record the last `StreamError.detail` onto `run.error`, persist it, and
  surface it from persisted state.
- Option B: have `agent_status` / `pending_agents` read the terminal StreamError
  directly from the run bus / in-memory run state.

## Decision

**Option A.** Two facts force it, so this is not an open user-facing fork:

1. Cross-process constraint. The MCP `agent_status` / `pending_agents` tools run
   in a SEPARATE process from the app and can read only PERSISTED state
   (`mcp_server.py` header comment, ~lines 130-136: "they read PERSISTED state:
   the AgentStore ... plus the backend's read_status"). The in-memory run bus and
   `Supervisor` are unreachable from that process. So option B ("read from the run
   bus/state") is infeasible for exactly the two tools the story names. The detail
   must be PERSISTED first - which is option A's job. Option B, to the extent it is
   realizable, DEPENDS on A; they are not real alternatives.

2. `pending_outcomes` semantics. It surfaces only WAITING / REPORTED / ERROR
   outcomes (`agent_store.py` ~862). For the story's "shows in pending_agents" to
   hold, the errored agent must be marked `AgentState.ERROR`, not left DONE.

## Shape

- `supervisor._drain`: on a `StreamError` event, set `run.error = event.detail`
  (last-wins). The supervisor RunPhase is deliberately LEFT unchanged (DONE on
  normal completion) - a StreamError is a normal terminal bus event, and the
  status/reattach endpoints already treat it as the run's end. The two lifecycle
  axes stay separate: RunPhase = "did the stream finish", AgentState = "did the
  turn succeed".
- `app.py persist`: the AGENT's terminal state is decided here. If
  `run_state.error` is set, mark `AgentState.ERROR` and use the detail as the
  durable outcome message (a captured reply text still wins when present). This
  also improves the pre-existing exception paths (CancelledError / stall / budget
  / generic), which set `run.error` but previously persisted an empty message.
- `mcp_server._agent_status_text`: read `store.outcome(agent_id)` and append an
  `error: <detail>` line for an ERROR outcome. `pending_agents` already renders
  `outcome.message`, so it needs no change.

## Consequence

An idle-timeout / over-limit / thread-setup StreamError now persists ERROR with a
diagnostic message, appears in `pending_agents`, and shows its detail in
`agent_status`. Behavior change to be aware of: a StreamError-terminated turn that
previously persisted as DONE (e.g. an idle timeout) now persists as ERROR - which
is the intended correction, not a regression.
