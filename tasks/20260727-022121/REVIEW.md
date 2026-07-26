# Review: Auto-delegate task implementation to a backend sub-agent (steering)

- TASK: 20260727-022121
- BRANCH: feature/delegate-agent-steer

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Two prompt-string edits in scufris/sessions.py (`_DELEGATION_CLAUSE` on the
orchestrator preamble; a work clause on the sub-agent preamble) plus three
tests. No BLOCKER/MAJOR/MINOR findings; two NITs, both no-change-needed.

Verification by the out-of-context reviewer, load-bearing claim re-derived in
session:

- Permission-mode claim (the crux of the delegation steer) verified in session:
  `enums.py:42` `MANUAL = "manual"  # read-only`; `create_agent` defaults
  `permission_mode="manual"` (mcp_server.py:449); `backends.py` maps manual ->
  codex `read-only` / claude `default` (headless has no approval answerer). So
  an implementing agent must get `edit`/`auto` - the steering text is correct
  and the reported "0 tool calls" stall is consistent with a read-only agent.
- All five delegation tool names exist verbatim as `@mcp.tool()` defs
  (list_projects, list_agents, create_agent, run_agent, agent_status), as do
  the comms tools it references. No dead names.
- Both preambles stay ONE `[scufris-tools]` block (open=1/close=1 each), so
  `strip_steering`'s count=1 cleans both roles.
- Tests non-vacuous: reverting `_DELEGATION_CLAUSE` fails
  `test_steer_orchestrator_gets_agent_delegation_chain`; reverting the work
  clause fails `test_steer_agent_told_to_implement_the_task_end_to_end`. The
  sub-agent `not in` assertions run against a genuine 630-char preamble that
  really lacks the delegation tools.
- Full QA gate green in the worktree: ruff, mypy (54 files), pytest.
- DECISION.md's backend-agnostic reasoning matches the code: the work clause
  mentions the flow skill only as an optional aid and the concrete steps stand
  alone (no hard dependency codex can't satisfy).

### NITs (no change made)

- [ ] R1.1 (NIT) scufris/sessions.py:121 - the clause says "the pending_agents
  / message_agent / acknowledge protocol above"; `_DELEGATION_CLAUSE` sits
  after `_COMMS_CLAUSE` in the block, so "above" is accurate. Implicit ordering
  dependency; no change required.
  - Response: Left as-is; "above" is correct given the clause order and the
    single-block composition is asserted by the single-block tests.
- [ ] R1.2 (NIT) tests/test_agent.py - the end-to-end assertion uses
  `"end-to-end" in lowered or "to completion" in lowered`; the preamble
  currently contains "end-to-end", so the `or` could rot silently if that
  phrasing were dropped. Low value.
  - Response: Left as-is; the `or` intentionally tolerates either phrasing so
    a future reword to "to completion" does not break the test spuriously.

### Pending manual check (operator's, not resolved by APPROVE)

- DoD #5: a live "implement task X using codex" turn (no tool names) makes the
  orchestrator create + run a write-capable agent that actually works the task
  and signals via request_input rather than finishing at 0 tool calls. Needs
  live codex/claude backends and writes to a project - operator acceptance,
  batched at the flow Finish.
