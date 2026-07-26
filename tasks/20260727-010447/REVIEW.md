# Review: macros food-lookup MCP tools

- TASK: 20260727-010447
- BRANCH: feature/macros-mcp-tools

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (in-session supplement: orchestrator-only scoping
  re-derived from the exact-equality test `test_apply_role_agent_keeps_only_request_input`)

The out-of-context reviewer ran the full gate green (ruff/mypy/532 passed) and
`nix flake check` green (pure sandbox, `macros` absent -> the `requires_macros`
real-CLI tests skip; argv/guard tests run). It independently verified: the argv
contract via mutation (changed `-q`->`-x`, `test_macros_argv_contract` FAILED as
required, restored); orchestrator-only scoping (`_AGENT_ROLE_TOOLS =
{"request_input"}`; `apply_role` strips all else; the exact-equality agent-role test
would fail if a macros tool leaked); temp-HOME hermeticity (the fixture's seeded DB
is used, not the operator's real csv - `_run` inherits the patched HOME); `add_food`
wrote only the temp DB (operator's real macros.csv, mtime Jul 8, has 0 rows matching
the test insert); and the lookup -> `journal_add_macros` chaining (lookup returns a
`what,protein,carbs,fat` row, exactly what `journal_add_macros` accepts).

- [ ] R1.1 (NIT) scufris/mcp_server.py:macros_lookup - a query that is EXACTLY a flag
  token ("-q", "-i", "-h") is consumed by the CLI's arg parser as a flag, not a food
  query. Not injection/arg-splitting (`_run` is shell=False + fixed argv; a
  multi-token "-q egg" is one argv element, treated as a food name); the CLI exits
  cleanly with a message `_run` surfaces. No real food is a bare flag.
  - Response: Left as-is (reviewer-recommended). Degenerate input only; the CLI
    already returns a clean message, and there is no `--` sentinel guaranteed by the
    CLI to harden against it.

- [ ] R1.2 (NIT) tests/test_mcp_server.py:test_macros_add_food_then_lookup_finds_it -
  the test also asserts the insert output echoes "oats"; a future silent-insert CLI
  would break that assertion though the subsequent lookup already proves the write.
  - Response: Left as-is. The echo assertion is a cheap extra signal; the lookup is
    the real proof, so a CLI change would surface clearly.

No BLOCKER/MAJOR/MINOR. No open `manual:` DoD items. APPROVE.
