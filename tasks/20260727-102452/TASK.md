# Add report_back sub-agent tool that wakes the orchestrator on completion

- STATUS: CLOSED
- PRIORITY: 60
- TAGS: feature,agents,mcp,wake

## Flow State

- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Goal

A sub-agent should be able to signal the orchestrator that it has FINISHED its
assigned work and hand back a result. Today a sub-agent role has exactly one
callback tool, `request_input` (`mcp_server.py:661`,
`_AGENT_ROLE_TOOLS = {"request_input"}`), which records a `WAITING` outcome the
orchestrator answers by resuming. But when a sub-agent finishes its work
naturally (a `DONE` outcome), nothing wakes the orchestrator and the agent never
appears in `pending_agents()` - the orchestrator can only find out by polling
`agent_status`. There is no opt-in "I am done, here is the result" signal.

After this change a sub-agent has a second callback tool, `report_back(summary)`,
that mirrors `request_input` but for COMPLETION: it records the summary against
the agent's current run, surfaces the agent in `pending_agents()` with a new
`reported` state, and wakes the orchestrator through the existing `WakeBridge`
(when `settings.auto_wake` is on). The orchestrator reads the report and
`acknowledge`s it - it does not need to resume the agent.

Observable done: a sub-agent turn that calls `report_back("implemented X; tests
green")` and ends makes that agent show in `pending_agents()` as
`STATE=reported` with the summary as its message, and (with `auto_wake` on)
grants the orchestrator a wake turn naming the reported agent; `acknowledge(id)`
clears it. A sub-agent that finishes WITHOUT calling `report_back` still records
a silent `DONE` and does NOT wake or surface (unchanged).

## Decision (see DECISION.md)

Two load-bearing shape choices, confirmed with the operator:

1. `report_back` records a NEW terminal state `AgentState.REPORTED` (value
   `"reported"`), not a reuse of `DONE` with a flag. This keeps the
   `pending_agents()` STATE column legible: `waiting` = needs a decision, resume
   and answer; `reported` = finished, read and acknowledge; `error` = crashed.
2. Wake behavior MIRRORS `request_input`: the push wake fires only when
   `settings.auto_wake` is on; the agent always surfaces in `pending_agents()`
   for the poll path. `report_back` does NOT force-wake when `auto_wake` is off.

## Design

`report_back` is a near-exact sibling of `request_input`. The signal is recorded
mid-run keyed to the current `run_id`, then preserved across the turn-end `DONE`
by `mark_finished`, so it survives the completion the same way a `WAITING` signal
does. The difference from `request_input` is the state (`REPORTED` vs `WAITING`)
and the intent (read+ack vs resume+answer). ERROR still wins over a preserved
signal (a crash after `report_back` is not a clean report).

Layers touched, in dependency order:

- `enums.py`: new `AgentState.REPORTED`.
- `agent_store.py`: `report_back()` store method (sibling of `request_input()`);
  generalize the `mark_finished` preserve logic from WAITING-only to
  WAITING-or-REPORTED; include REPORTED in `pending_outcomes()`.
- `app.py`: `AgentReportBack` / `ReportBackResult` models; `POST
  /api/agents/{id}/report_back` endpoint (sibling of `agent_request_input`);
  `PendingAgent` doc updated to mention REPORTED.
- `wake.py`: `WakeBridge.on_run_complete` fires on REPORTED too; `wake_prompt`
  labels each agent's state so a reported agent reads "finished - read and
  acknowledge" rather than "answer and resume".
- `mcp_server.py`: `report_back(summary)` tool; add `"report_back"` to
  `_AGENT_ROLE_TOOLS`; `pending_agents()` docstring mentions the reported state.
- `sessions.py`: extend `AGENT_STEERING_PREAMBLE` (same single block) with a
  clause steering the sub-agent to call `report_back(summary)` when it FINISHES,
  instead of ending silently. Codex only honors tool-choice on the turn prompt
  (`codex-tool-choice-only-steers-via-the-turn-prompt`), so the tool needs this
  steer to actually get used.
- `web/src/style.css`: a `.agents__badge--reported` color (the badge text is the
  state string already, so only CSS is needed).

## Steps

- [x] `enums.py`: add `REPORTED = "reported"` to `AgentState` with a comment
      ("ended a turn having reported its result; orchestrator reads + acks"),
      placed next to `WAITING`.
- [x] `agent_store.py`: add `report_back(agent_id, summary, *, run_id="",
      session_id=None) -> RunOutcome`, a sibling of `request_input()` that writes
      a `RunOutcome(state=REPORTED, message=summary, run_id, session_id,
      acknowledged=False)` after the `_raw` existence guard.
- [x] `agent_store.py`: generalize `mark_finished`'s `preserve_waiting` to cover
      REPORTED. Rename to `preserve_signal` (or keep the name and broaden the
      predicate) so a same-run, unacknowledged WAITING **or** REPORTED outcome is
      preserved through the turn-end DONE, refreshing only session_id/ts, and
      `eff_state` is the preserved state. ERROR and a later run's completion
      still overwrite it. Update the method docstring.
- [x] `agent_store.py`: include `AgentState.REPORTED` in `pending_outcomes()`'s
      state filter and update its docstring.
- [x] `app.py`: add `AgentReportBack(summary: str)` and
      `ReportBackResult(agent_id, state)` models (siblings of the request_input
      pair); update `PendingAgent`'s comment to include REPORTED.
- [x] `app.py`: add `POST /api/agents/{agent_id}/report_back` handler
      (`agent_report_back`), mirroring `agent_request_input`: `_require_agent`,
      422 on empty summary, call `agents.report_back(agent_id, summary,
      run_id=agent_runs.get(agent_id, ""), session_id=agent.session_id)`, 404 on
      `AgentNotFound` (the orchestrator is not a sub-agent).
- [x] `wake.py`: `WakeBridge.on_run_complete` enqueues on
      `state in (WAITING, ERROR, REPORTED)`; `wake_prompt` renders each agent's
      state and tailors the closing instruction (waiting/error -> answer via
      message_agent then acknowledge; reported -> read the report then
      acknowledge, no resume needed). Keep the batch fold behavior.
- [x] `mcp_server.py`: add `report_back(summary: str)` tool (sibling of
      `request_input`): resolve `_self_agent_id()`, POST
      `/api/agents/{id}/report_back` with `{"summary": ...}`, same empty/no-id
      guards. Docstring: call this when you have FINISHED the task, pass a short
      result summary, then END your turn; the orchestrator will be woken / see it
      in pending_agents. Add `"report_back"` to `_AGENT_ROLE_TOOLS`. Update the
      `pending_agents()` tool docstring and the module header comment (line ~15)
      to mention the reported state.
- [x] `sessions.py`: extend `AGENT_STEERING_PREAMBLE` inside its SINGLE
      `[scufris-tools]` block with a completion clause: when you have carried the
      task to completion, call `report_back(summary)` with a short result and
      STOP, rather than ending silently, so the orchestrator knows you finished.
      Keep it backend-agnostic and one block; do not add a second block
      (`strip_steering` removes only the leading block).
- [x] `web/src/style.css`: add `.agents__badge--reported` (a distinct color,
      e.g. `var(--blue)`/`var(--cyan)` family, distinct from done's green and
      waiting's) next to the existing badge rules (~line 2050).
- [x] Tests (see Definition of Done for the exact proofs).
- [x] Verify: `ruff check .`, `mypy .`, `python -m pytest`, and the web build /
      `npm test` in `web/` all green.

## Definition of Done

- [x] `AgentState.REPORTED` exists with value `"reported"`.
      (test: tests/test_app.py or a store test)
- [x] `POST /api/agents/{id}/report_back` records a REPORTED outcome carrying the
      summary, returns `{agent_id, state: "reported"}`; 422 on empty summary; 404
      on unknown agent and on the orchestrator.
      (test: tests/test_app.py, sibling of the request_input endpoint tests)
- [x] After `report_back`, the agent shows in `GET /api/agents/pending` with
      `state="reported"` and `message=summary`, and `acknowledge` drops it (and is
      idempotent). A cleanly-DONE agent that did NOT report is still absent from
      pending. (test: tests/test_app.py, extend the pending/acknowledge roundtrip)
- [x] `mark_finished` preserves a same-run unacknowledged REPORTED outcome
      through the turn-end DONE (not clobbered), and an ERROR or a later run still
      overwrites it - the exact preserve invariants the WAITING path already has.
      (test: a store test asserting the preserved state, mirroring the existing
      preserve_waiting test)
- [x] `WakeBridge` wakes the orchestrator on a REPORTED completion when
      `auto_wake` is on (and not when off), batching alongside WAITING/ERROR, and
      `wake_prompt` names the reported agent with read+ack guidance.
      (test: tests/test_wake.py, mirroring the WAITING wake test)
- [x] The MCP `report_back` tool posts `{"summary": ...}` to
      `/api/agents/{env-agent}/report_back`; `report_back` is exposed to the agent
      role and NOT the orchestrator role; `request_input` is unaffected.
      (test: tests/test_mcp_server.py, sibling of the request_input tool test +
      a role_tool_names assertion)
- [x] `AGENT_STEERING_PREAMBLE` names `report_back` for the finish case AND still
      names `request_input` for the blocked case, stays ONE `[scufris-tools]`
      block, and `strip_steering` round-trips it. The orchestrator
      `STEERING_PREAMBLE` does NOT gain `report_back`.
      (test: tests/test_agent.py, extend the preamble tests)
- [x] Full QA gate green.
      (cmd: `python -m pytest`; `ruff check .`; `mypy .`; web build/tests)

## Non-goals

- No new frontend polling/notification UI for reported agents beyond the badge
  color; the orchestrator surface (pending_agents / wake) is the consumer.
- No change to `request_input`, `acknowledge`, or the WAITING semantics beyond
  generalizing the preserve predicate.
- No change to the default `auto_wake=off`; report_back honors it exactly like
  request_input.

## Implementation Notes

Implemented as a near-exact sibling of `request_input` across seven files, in
dependency order (enums -> store -> app -> wake -> mcp_server -> sessions -> css):

- `enums.py`: `AgentState.REPORTED = "reported"` next to `WAITING`.
- `agent_store.py`: `report_back()` mirrors `request_input()` (existence guard,
  then a REPORTED `RunOutcome`). `mark_finished`'s `preserve_waiting` predicate
  was generalized to `preserve_signal` (WAITING **or** REPORTED, same-run,
  unacknowledged), and `eff_state` now carries the preserved state rather than a
  hard-coded WAITING - so a DONE turn-end keeps a REPORTED signal, while ERROR and
  a later run still overwrite it. `pending_outcomes()` includes REPORTED.
- `app.py`: `AgentReportBack`/`ReportBackResult` models and the
  `POST /api/agents/{id}/report_back` handler, mirroring the request_input pair
  (422 empty, 404 unknown/orchestrator). The route auto-tags `agents` via
  `_route_tags` (path prefix), so the OpenAPI all-routes-tagged test passes with
  no change.
- `wake.py`: the wake batch changed from `dict[str, str]` to
  `dict[str, tuple[AgentState, str]]` so `wake_prompt` can label each agent's
  state and tailor guidance (waiting/error -> resume+answer; reported -> read the
  report, no resume). `on_run_complete` enqueues WAITING/REPORTED/ERROR.
- `mcp_server.py`: `report_back(summary)` tool; `"report_back"` added to
  `_AGENT_ROLE_TOOLS`; `pending_agents()` docstring + module header updated.
- `sessions.py`: `AGENT_STEERING_PREAMBLE` gained a third clause (finish ->
  `report_back(summary)`) inside its single `[scufris-tools]` block.
- `web/src/style.css`: `.agents__badge--reported` uses `var(--yellow)` (the
  palette has cyan=running, amber=queued, green=done, red=error; yellow reads as
  "successful but needs your attention"). The badge text is already the state
  string, so no TS change (`state: string`, not a union).

Verification: `ruff check .`, `mypy .` (54 files), `python -m pytest` (all pass),
`npm run build` + `npm test` (186 web tests) all green in the nix devshell.

The one manual DoD (a live delegated codex/claude run actually calling
report_back) is left for the operator to confirm against a real backend, same as
the sibling steering task 20260727-022121 - it needs live backends, not CI.
