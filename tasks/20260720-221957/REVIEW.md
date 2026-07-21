# Review: A5 orchestrator observation MCP tools

- TASK: 20260720-221957
- BRANCH: feature/orchestrator-tools

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Check suite (reviewer ran it in the worktree): ruff clean, mypy clean (35
files), `python -m pytest` = 250 passed. Verified independently in-session.

Reviewer verified: the cross-process design is sound (each helper builds a fresh
AgentStore that loads agents.json from disk + read_status reads the rollout/
session files - the same persisted state the app writes); read_status failure is
wrapped so the tool never crashes; a missing session_id -> no progress lines, no
error; unknown id -> a clear "no such agent" message; genuinely read-only (only
list/get/read_status); `test_tools_registered` includes both new tools (not a
stale set); the status test proves a real cross-process re-read after
mark_finished; lazy imports + TYPE_CHECKING keep MCP startup light and mypy
clean; close-out matches the code.

- [x] R1.1 (MINOR) mcp_server.py `list_agents` docstring enumerated states as
  "idle/running/done/error", omitting "blocked" (AgentLifecycle has five). A
  model could mislabel a blocked agent.
  - Response: Fixed. The docstring now lists idle/running/blocked/done/error.
- [ ] R1.2 (NIT) `_list_agents_text` truncates the fixed-width columns but leaves
  the trailing `name` un-capped; the whole output is separately capped by the
  server's output limit, so a long name can only wrap, not overflow.
  - Response: Left as-is - it is the trailing column and the server caps total
    output; truncating a name would lose information for no real benefit.

## Round 2

- VERDICT: APPROVE
- REVIEWER: in-session (a one-line docstring correction; no code/behavior change)

Verification: the `list_agents` docstring now enumerates all five lifecycle
states. Suite re-run: ruff + mypy clean, 250 passed. No new findings.
