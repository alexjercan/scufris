# Decision: A user-cancelled run is a distinct CANCELLED state, not reuse of ERROR

- DATE: 20260728-134840
- STATUS: ACCEPTED
- TASK: 20260728-134840
- TAGS: decision, agents, backend, ui, streaming

## Context

The run engine already funnels `asyncio.CancelledError` into
`run.error = "cancelled"` and, on the persist path, an agent with a run error
becomes `AgentState.ERROR` with the detail as its message (commit 0b2bd43).
Adding a user-facing cancel (stop button + orchestrator `cancel_agent` tool)
means cancellation stops being an internal shutdown event and becomes a normal,
intentional user action. The `AgentState` enum today has no neutral terminal
state for "the user stopped this on purpose"; the only stop states are `DONE`,
`ERROR`, and the waiting/reported states. So the fork is: reuse `ERROR` with a
"cancelled" message, or introduce a distinct `CANCELLED` state.

These are mutually exclusive because the ERROR state is load-bearing elsewhere:
`pending_agents` / `pending_outcomes` surface ERROR agents to the orchestrator
as things needing attention, and the UI paints ERROR as a failure. A
user-initiated stop must NOT read as a crash needing the orchestrator, so it
cannot share the ERROR state without corrupting those consumers.

## Decision

Introduce `AgentState.CANCELLED` as a distinct terminal state. Cancellation is
signalled by an explicit `run.cancelled` flag on the supervised run (not by
matching the `run.error == "cancelled"` string), threaded through the run
snapshot to the persist callback, which maps it to `CANCELLED`. `CANCELLED`
outcomes are excluded from the pending set. Partial assistant output produced
before the cancel is kept in the transcript, marked interrupted.

## Alternatives considered

- **Reuse ERROR("cancelled")** - already half-wired, zero enum churn. Rejected:
  a deliberate user stop would render as a red failure and would surface in
  `pending_agents` as an agent needing the orchestrator, which is wrong. It also
  keys behaviour off a magic error string, so a genuine backend error whose
  detail happens to be "cancelled" would be misclassified.
- **No terminal state change; only close the SSE relay (frontend abort only)** -
  simplest, no backend change. Rejected: the backend turn keeps running as a
  supervised job, so tokens keep being generated and billed and the subprocess
  lives on. That is a detach, not a cancel, and fails the user's ask to actually
  stop the run.

## Consequences

- Easier: the UI and orchestrator can treat a stop as neutral; pending logic
  stays honest; the Claude subprocess is truly killed via the generator finally.
- Harder: `CANCELLED` is a new route into terminal state that every
  `AgentState` consumer (status text, pending logic, UI badges, telegram) must
  handle - the plan carries a grep-audit step for that. Persistence
  (`outcomes.json`) gains a new enum value; old records without it still load
  because `AgentState` is a StrEnum read leniently.
