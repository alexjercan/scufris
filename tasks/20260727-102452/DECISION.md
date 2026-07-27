# DECISION: report_back outcome shape and wake behavior

STATUS: ACCEPTED (operator confirmed at the plan gate, 20260727)

## Context

Adding a `report_back` sub-agent callback tool so a sub-agent can signal the
orchestrator that it FINISHED (vs `request_input`, which signals it is BLOCKED).
Two shape choices are load-bearing because they change the observable pending
surface and the wake semantics, so they went to the operator before building.

## Decision 1: a new AgentState.REPORTED, not reuse of DONE + a flag

`report_back` records `AgentState.REPORTED` (value `"reported"`), a distinct
terminal state, rather than writing `DONE` with a `reported`/wants-wake flag on
`RunOutcome`.

Rationale: the `pending_agents()` STATE column must stay legible. With a distinct
state the orchestrator reads it directly - `waiting` = resume and answer,
`reported` = read and acknowledge, `error` = crashed. A DONE+flag encoding hides
the distinction behind a boolean the STATE column cannot show, and would risk a
silent DONE and a reported DONE looking identical to any reader that keys off
state alone.

Cost accepted: one new enum member plus a `.agents__badge--reported` CSS color;
the badge text is already the state string, so no rendering-code change.

## Decision 2: wake mirrors request_input (auto_wake-gated), not force-wake

`report_back` wakes the orchestrator through the existing `WakeBridge` ONLY when
`settings.auto_wake` is on; it always surfaces the agent in `pending_agents()`
for the poll path. It does NOT force a wake when `auto_wake` is off.

Rationale: consistency with `request_input`, which already behaves this way, and
avoids a completion barging into an unrelated orchestrator turn against the
operator's `auto_wake=off` default. The poll path (`pending_agents`) remains the
always-available channel, exactly as for blocked agents.

## Consequences

- `mark_finished`'s preserve logic generalizes from WAITING-only to
  WAITING-or-REPORTED (same-run, unacknowledged, DONE-does-not-clobber; ERROR and
  later runs still win).
- `pending_outcomes()` and the `WakeBridge` filter both add REPORTED.
- Sub-agent steering (`AGENT_STEERING_PREAMBLE`) must name `report_back` for the
  finish case, or codex will not call it
  (`codex-tool-choice-only-steers-via-the-turn-prompt`).
