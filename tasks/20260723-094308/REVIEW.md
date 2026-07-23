# Review: BC3 pending_agents + acknowledge orchestrator MCP tools

- TASK: 20260723-094308
- BRANCH: feat/pending-agents

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (round-1 findings), in-session pass verified and recorded

The out-of-context reviewer ran the full suite (367 passed, ruff + mypy clean),
ran each DoD proof by name, and genuinely probed the two risk areas: it RENAMED
the static `/api/agents/pending` route and confirmed
`test_pending_agents_and_acknowledge_roundtrip` then fails (so the test exercises
the real route, not a 404-that-looks-empty), and it mutated the pending predicate
to include DONE and confirmed `test_pending_outcomes_lists_waiting_and_error_only`
fails. In session I re-derived the route-ordering claim: the round-trip asserts
`len(pending) == 1` with real fields after `request_input` - a shadowed
`/pending` would resolve `get_agent("pending")` -> 404, whose body is not a
1-element list, so the assertion genuinely proves the ordering. No BLOCKER/MAJOR/
MINOR.

- [ ] R1.1 (NIT) scufris/mcp_server.py `acknowledge` - returns the raw JSON body
  to the model rather than a rendered line like `pending_agents`. Consistent with
  the other POST tools (`request_input`, `message_agent`), so cosmetic only.
  - Response: left as-is, per the reviewer - it matches the POST-tool convention;
    rendering would be the inconsistency.

No open `manual:` DoD items for this task (all proofs are `test:`/`cmd:`).
</content>
