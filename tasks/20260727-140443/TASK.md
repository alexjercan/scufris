# surface run.error / backend StreamError detail through agent_status and pending_agents

- PRIORITY: 1
- TAGS: backlog, agent, orchestrator, dx
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As an operator watching a delegated sub-agent, I want `agent_status` /
`pending_agents` to report WHY a run ended in error, so a backend `StreamError`
(idle timeout, over-limit line, thread-setup failure) shows a diagnostic message
instead of an empty one.

## Context

Split out of `20260727-133302` (>64 KiB app-server line fix). A backend-level
`StreamError` is published to the run bus and the run ends in `DONE` state
(`supervisor.py` `_drain` only publishes events; it does not set `run.error` for
a StreamError event). The orchestrator's `agent_status` reports `read_status`
fields, not the last `StreamError`, so any backend error (not just the
over-limit one) surfaces to the orchestrator as "error with no message". This is
a general orchestrator-visibility gap, independent of any single backend bug.

## Surface decision (see DECISION.md)

Option A (record `StreamError.detail` onto `run.error`, persist it, surface from
persisted state) is the ONLY viable surface: the MCP `agent_status` /
`pending_agents` tools run in a SEPARATE process and can read only PERSISTED
state, never the in-memory run bus (`mcp_server.py` header, lines ~130-136), so
option B ("read the terminal StreamError from the run bus") is infeasible
cross-process. The errored agent is marked `AgentState.ERROR` (not left DONE) at
the persist chokepoint, because `pending_outcomes` only surfaces
WAITING/REPORTED/ERROR - the story's "shows in pending_agents" cannot hold
otherwise. Recorded in DECISION.md.

## Steps

- [x] `supervisor.py` `_drain`: when publishing a `StreamError` event, record
      `run.error = event.detail` (last-wins). Run flow unchanged (RunPhase stays
      DONE on normal stream completion; snapshot already carries `error`).
- [x] `app.py` `_launch_agent_turn.persist`: derive the agent's terminal
      state/message from `run_state.error`. When it is set (exception paths OR a
      StreamError event), mark `AgentState.ERROR` with the detail as the durable
      outcome message (captured reply text wins if present). Clean DONE unchanged.
- [x] `mcp_server.py` `_agent_status_text`: read the persisted outcome
      (`store.outcome(agent_id)`) and append an `error: <detail>` line when the
      outcome is ERROR / carries an error message. `pending_agents` needs no
      change (it already renders `outcome.message`).
- [x] Test (supervisor): a stream yielding a terminal `StreamError(detail=...)`
      leaves `sup.status(id).error == detail`, state `"done"`.
- [x] Test (app, end to end): a regular agent run whose backend yields a
      `StreamError` persists state `"error"` and `/api/agents/pending` carries the
      detail as message.
- [x] Test (mcp): an ERROR outcome makes `_agent_status_text` show `state: error`
      and `error: <detail>`.

## Definition of Done

- A backend StreamError-terminated run persists `AgentState.ERROR` with the
  detail as its outcome message (test: new agent-run test in `tests/test_app.py`).
- `agent_status` surfaces the detail (test: new `tests/test_mcp_server.py`
  assertion on the `error:` line; cmd: `git diff master... -- scufris/mcp_server.py
  | grep -n "error:"` shows the new line).
- `pending_agents` / `/api/agents/pending` lists the errored agent with the detail
  (test: the test_app pending assertion).
- The supervisor records `StreamError.detail` on `run.error` (test: new
  `tests/test_supervisor.py` test).
- Full QA gate green (cmd: `nix flake check`).

## Notes

- Files: `scufris/supervisor.py` (`_drain` ~318, `_execute` error handling
  ~277-295, `_Run.snapshot` ~120-127); `scufris/app.py`
  (`_launch_agent_turn.persist` ~1224); `scufris/mcp_server.py`
  (`_agent_status_text` ~160, `pending_agents` ~510); `scufris/agent_store.py`
  (`pending_outcomes` ~862, `outcome` ~852, `mark_finished` ~717).
- `supervisor.start` has exactly ONE caller (`_launch_agent_turn`), so `persist`
  is the single terminal chokepoint for every supervised turn.
- Affects every backend `StreamError` equally (idle-timeout path
  `agent.py` included), so fixing it here helps all error paths, not just the
  over-limit line.
