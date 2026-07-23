# Review: SC2 orchestrator comms steering

- TASK: 20260723-153615
- BRANCH: feat/sc2-orchestrator-comms-steering
- DATE: 20260723
- REVIEWER: out-of-context agent (round 1)
- VERDICT: APPROVE

## Round 1 - VERDICT: APPROVE

Reviewed the tip commit against the actual code (not just the diff). Targeted
tests (`tests/test_agent.py tests/test_sessions.py`) green.

### Findings

1. [verified-ok] Single sentinel block: `STEERING_PREAMBLE` is
   `open + host clause + comms clause + close` - exactly one `[scufris-tools]` /
   `[/scufris-tools]` pair, no stray sentinels in either clause body, so
   `strip_steering` (regex, `count=1`) fully removes it.
2. [verified-ok] Orchestrator-only routing: `_steer` returns `STEERING_PREAMBLE`
   for `is_orchestrator`, `AGENT_STEERING_PREAMBLE` for `agent_id`, bare prompt
   otherwise. `pending_agents`/`message_agent`/`acknowledge` live only in the comms
   clause, so they cannot leak into a sub-agent's preamble.
3. [verified-ok] Tool names/signatures in the steering text match the real MCP
   tools exactly: `pending_agents()`, `message_agent(agent_id, message)`,
   `acknowledge(agent_id)`. No drift, correct arg order.
4. [verified-ok] `test_steer_orchestrator_gets_comms_protocol` asserts both
   directions: orchestrator gets all three tool names; a sub-agent turn gets none;
   `strip_steering` still yields the clean user text.
5. [verified-ok] Pure ASCII, no comment-vs-code drift.

### Nit (non-blocking)
- The older module comment line ("Both blocks share the same sentinel wrapping so
  strip_steering cleans either one") slightly undersells that the orchestrator
  block now carries two clauses; the paragraph directly below clarifies it. No
  change requested.

No blockers or majors. Ship it.
