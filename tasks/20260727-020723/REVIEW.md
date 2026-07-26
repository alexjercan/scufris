# Review: Orchestrator logs food from plain language (steer STEERING_PREAMBLE)

- TASK: 20260727-020723
- BRANCH: feature/orchestrator-food-logging-steer

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

No BLOCKER/MAJOR/MINOR/NIT findings. The diff adds `_JOURNAL_CLAUSE` as the
third clause of the single `STEERING_PREAMBLE` block (scufris/sessions.py) and
two tests; nothing else changed.

Verification performed by the out-of-context reviewer and re-confirmed in
session:

- Full QA gate green in the worktree: `ruff check .` (clean), `mypy .` (no
  issues, 54 files), `python -m pytest` (all pass). Re-run in session this
  cycle.
- Tests are non-vacuous: reverting `_JOURNAL_CLAUSE` makes
  `"macros_lookup" in steered` / `"journal_add_macros" in steered` both False,
  so `test_steer_orchestrator_gets_journal_food_chain` would fail (observed
  test-first red before the clause was added). The sub-agent `not in`
  assertions are meaningful - `AGENT_STEERING_PREAMBLE` is non-empty and
  genuinely lacks these tool names.
- Single-block invariant preserved:
  `STEERING_PREAMBLE.count("[scufris-tools]") == 1` and one closing marker;
  `strip_steering` round-trips an orchestrator-steered prompt back to the raw
  text (`orchestrator-steering-is-one-block-two-clauses`).
- All 11 tool names in the steering text exist verbatim as `@mcp.tool()` defs
  in scufris/mcp_server.py (`ground-steering-text-in-the-real-tool-signatures`).
- Load-bearing claim re-derived: `macros_lookup` returns
  `<food> <amount><unit>,<protein>,<carbs>,<fat>` (e.g. `egg 2pc,12,0,10`),
  which is exactly the CSV row `journal_add_macros(row)` accepts, so the two
  chain with no reshaping.
- Implementation Notes match the code; no scope creep.

### Pending manual check (operator's, not resolved by APPROVE)

- DoD #4: a live codex turn whose only user text is "log that I had 2 eggs"
  (no tool names) drives `macros_lookup` then `journal_add_macros` and reports
  the updated daily total, against the real the-den + macros DB. This needs a
  real codex turn that WRITES to the operator's journal, so it is human
  acceptance, batched at the flow Finish.
