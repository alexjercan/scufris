# Review: CRUD control MCP tools for projects and agents

- TASK: 20260722-232723
- BRANCH: feature/crud-control-tools

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Correct and faithful to the spec. The reviewer cross-checked `_provided(...)` against
the real `ProjectUpdate` / `AgentUpdate` models (both `extra="forbid"` + all-optional,
so omitting None fields is right; an explicit `""` is legitimately kept), confirmed the
empty-body guard fires before any HTTP call, confirmed `_clean_id` guards every id and
`_api_call` never raises, and confirmed the registration set is the exact 15-tool set.
Tests assert method+path+body via side_effect handlers (fail on revert); the
orchestrator-rejection test registers no respx route, so a leaked HTTP call would error.
Full pytest green, mypy clean on the changed file.

Re-verified in-session: in BOTH `update_agent` and `delete_agent`, the order is
`_clean_id` -> `_reject_orchestrator` -> `_api_call`, so the reserved orchestrator is
refused before any HTTP call (the user's "regular agents only" scope holds).

No BLOCKER/MAJOR/MINOR/NIT findings.

Open `manual:` DoD items: none.
