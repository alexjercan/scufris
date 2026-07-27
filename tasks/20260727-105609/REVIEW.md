# Review: Split MCP into scufris + den + agent servers; per-server live health

- TASK: 20260727-105609
- BRANCH: feature/split-mcp-den

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Both gates pass independently: the Python gate (`ruff check . && mypy scufris/
&& python -m pytest -q`) and the web gate (`npm run ci` - format:check + eslint +
vitest + webpack build) are green. All DoD-named proofs run and pass
(`test_orchestrator_registers_scufris_and_den`,
`test_subagent_registers_only_callback_server`,
`test_servers_expose_disjoint_tool_sets`,
`test_mcp_health_den_warn_when_unconfigured`, `test_mcp_health_marks_disabled`,
`test_agent_tools_lists_the_mcp_tools`). Tool census confirmed live: scufris 17 /
den 12 / agent 2 = 31, disjoint. The in-session pass independently re-derived the
load-bearing isolation claim: the agent-turn call site (`app.py:1312`,
`is_orchestrator=agent.id == ORCHESTRATOR_ID`) plus `scufris_mcp_servers`
branching (is_orchestrator wins; a sub-agent gets only the `agent` server) means
a sub-agent turn never registers the orchestrator/den servers - a physical
guarantee, re-confirmed by re-running the registration + disjoint tests.

Only three NIT findings (no BLOCKER/MAJOR/MINOR), so the verdict is APPROVE; the
NITs are addressed below as cheap doc/naming-accuracy fixes.

- [x] R1.1 (NIT) scufris/agent.py (`_stream_app_server` docstring) - the docstring
  still described the retired "orchestrator role" / "agent role (only the
  request_input callback)" model and omitted `report_back`. Reword to the
  physical audience-split model and mention both callbacks.
  - Response: fixed - docstring rewritten to the audience/physical-split model, both
    callbacks named.
- [x] R1.2 (NIT) scufris/agent.py:317 (`_steer` docstring) - opening line still said
  "Prepend the role's tool-steering preamble"; the body is already updated. Change
  "role's" to "audience's".
  - Response: fixed.
- [x] R1.3 (NIT) tasks/.../TASK.md DoD - the tools-tagging proof is named
  `test_agent_tools_lists_the_mcp_tools`, but the DoD calls it
  `test_agent_tools_tagged_by_server`. Rename the test to the DoD name (or note the
  mapping) so the proof is greppable.
  - Response: fixed - renamed the test to `test_agent_tools_tagged_by_server` to match the DoD.

### Pending user check (manual DoD)

- manual: On the running dashboard, the orchestrator settings show `scufris`
  (green) and `den` (green when the-den is configured, amber otherwise) with
  per-tool bulbs, and a sub-agent's settings show only the callback server. The
  reviewer verified the data path, endpoints, render logic and tests but did not
  launch the live dashboard - to be confirmed by the operator.
