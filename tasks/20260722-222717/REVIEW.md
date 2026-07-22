# Review: T1 - orchestrator-only scufris MCP scoping

- TASK: 20260722-222717
- BRANCH: feature/orchestrator-only-mcp

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

The branch delivers the Goal: the built-in scufris MCP server AND its steering
preamble are both gated on `is_orchestrator`, consistently, at the argv-composition
seam. Full suite green in the nix dev shell (ruff check + ruff format --check +
pytest all pass); the 4 changed source files add zero mypy errors (the 44 red are
pre-existing, task 20260720-174021). Both DoD tests run and pass individually; the
argv test is a genuine real-spawn proof (fake codex dumps its own argv). The two
`backend.stream` call sites are `cli.py:71` (orchestrator, True) and `app.py:1106`
(`agent.id == ORCHESTRATOR_ID`) - re-verified in-session. Landing chat / fork /
resume paths all resolve the orchestrator agent, so it keeps its tools on resumed
turns and regular agents never receive them. CHANGELOG + TASK notes match the code;
doc sweep found no stale "every agent gets the tools" claims.

- [ ] R1.1 (NIT) scufris/backends.py:144 - `is_orchestrator` defaults to `False`
  on the `AgentBackend.stream` Protocol; consider making it required so a future
  backend/caller can't silently omit it and lose tools. Optional.
  - Response: Declined by design. The `False` default is deliberately fail-closed:
    a caller who forgets the flag gets NO scufris tools (safe) rather than
    accidentally leaking the orchestrator-only tools to a regular agent, and
    `test_mcp_overrides_scopes_scufris_to_orchestrator` pins that default. A
    required kwarg would also be inconsistent with the other keyword-only,
    defaulted params on the same signature (`session_id`, `cwd`, ...). Left as-is.

Open `manual:` DoD items: none (T1's DoD is all test:/cmd:).
