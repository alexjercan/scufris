# Retro: surface backend StreamError detail via agent_status / pending_agents

## What shipped

A backend that ends a turn with a terminal `StreamError` (idle timeout,
over-limit line, thread-setup failure) now surfaces WHY through the orchestrator
tools:

- `supervisor._drain` records the last `StreamError.detail` on `run.error`
  (RunPhase left DONE - a StreamError is a normal terminal bus event).
- `app.py` persist (the single terminal chokepoint) marks the agent
  `AgentState.ERROR` whenever `run_state.error` is set, using the detail as the
  durable outcome message. The error detail wins over any captured reply.
- `mcp_server._agent_status_text` reads the persisted outcome and appends a
  flattened/capped `error: <detail>` line; `pending_agents` already rendered
  `outcome.message`.
- `pending_outcomes` now excludes the reserved orchestrator (it is the consumer
  of that poll, never a member).

Tests: supervisor records the detail; an end-to-end agent run whose backend
yields a StreamError (and one that RAISES) persists ERROR with the detail and
shows it in `/api/agents/pending`; `agent_status` renders the `error:` line; a
clean DONE shows none; the orchestrator is excluded from pending.

## What went well

- The design "fork" in the task's Step 1 dissolved once the code was read: the
  MCP tools are a separate process that can only read PERSISTED state, so option
  B (read the run bus) was infeasible and option A was forced. Reading the
  `mcp_server.py` header comment up front turned a would-be user question into a
  DECISION.md with the constraint named. No blind guessing, no user round-trip.
- Tracing the ACTUAL terminal flow (StreamError -> stream completes normally ->
  RunPhase DONE -> persist marks DONE with empty message) revealed the real
  regression: after the parent fix, a failed turn looked SUCCESSFUL (DONE), not
  just "error with no message". The story's framing was slightly stale; the code
  was the authority.
- Confirming `supervisor.start` had exactly ONE caller made "persist is the
  single chokepoint" a fact, not a hope - so the fix landed in one place.

## What went wrong / difficulties

- First end-to-end test asserted on `/api/agents/{id}/status`, which reports the
  live supervisor RunPhase (DONE), not the persisted AgentState (ERROR). It went
  red with `done != error`. The fix was to poll the durable record
  (`/api/agents/{id}`) instead: the in-scope operator surfaces read PERSISTED
  state, and `/status` deliberately mirrors the run lifecycle. Lesson below.
- The persist message precedence had a latent edge (StreamDone-then-StreamError:
  a captured reply masking the error detail) that the out-of-context reviewer
  caught (R1.1). The initial "captured wins" felt natural but is wrong for a
  FAILED run; the reviewer's "error detail wins on failure" is clearly right.
- Marking the agent ERROR made the ORCHESTRATOR itself eligible to appear in its
  own `pending_agents` poll - a new surface I introduced without noticing.
  Reviewer flagged it (R1.5); filtered it in `pending_outcomes`.

## What to do differently next time

- When a persisted-state change alters an agent's terminal STATE, immediately
  ask "who reads this state, and does any consumer treat the reserved
  orchestrator specially?" `list()` hides the orchestrator; `pending_outcomes`
  did not, and the new ERROR path made that gap reachable. Enumerate the
  consumers of a state field before flipping it.
- For an assertion about a run's terminal outcome, poll the DURABLE record, not
  the live `/status` view: `/status` reports RunPhase (the run lifecycle), which
  can legitimately differ from the persisted AgentState (the turn's success).
  These are two independent axes by design.

## Lessons to fold into LESSONS.md

- `status-endpoint-reports-runphase-not-persisted-agentstate`: `/api/agents/{id}/
  status` returns the live supervisor RunPhase when a run record exists (else the
  persisted state), so a StreamError-terminated turn reads `done` there while the
  persisted AgentState is `error`. Assert a turn's terminal OUTCOME on the durable
  record (`/api/agents/{id}` or the OutcomeStore), not `/status`. RunPhase (did
  the stream finish) and AgentState (did the turn succeed) are independent axes.
- `flipping-a-terminal-state-needs-a-consumer-and-reserved-member-sweep`: when a
  change makes an agent reach a terminal STATE it could not before (here ERROR via
  a StreamError), sweep every reader of that state - especially ones that treat
  the reserved orchestrator specially. `list()` hid the orchestrator but
  `pending_outcomes` did not, so the orchestrator could self-appear in its own
  "who needs me" poll until the exclusion was added. Caught by out-of-context
  review.
- `error-outcome-message-beats-a-captured-reply`: on a FAILED turn the durable
  outcome message must be the failure detail, not a stale captured success reply
  (a rogue backend can emit a done frame then a trailing StreamError). Prefer
  `run_state.error` over the captured reply when the run failed. Caught by
  out-of-context review (R1.1).
</content>
