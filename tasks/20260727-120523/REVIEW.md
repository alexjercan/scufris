# Review: Move MCP per-server health into the Health section; dropdown organizational

- TASK: 20260727-120523
- BRANCH: feature/mcp-health-in-health-section

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Both gates pass independently: the Python gate (`ruff check . && mypy scufris/ &&
python -m pytest`) and the web gate (`npm run ci` - format:check + eslint + vitest
+ webpack build) are green (188 web tests + build). The reviewer found no
BLOCKER/MAJOR/MINOR/NIT against the diff. The in-session pass independently
re-verified the load-bearing claim by running the three DoD tests by full id:
`test_agent_health_mcp_rows_are_per_server_and_audience_aware`,
`test_agent_health_den_row_warns_when_unconfigured`, and
`test_agent_health_endpoint_reports_checks` - all pass (audience-aware rows:
orchestrator -> `mcp: scufris` + `mcp: den`; sub-agent -> `mcp: agent`; no-scufris
backend -> single "none" row; den amber when unconfigured).

What the reviewer verified (re-confirmed in-session): `_mcp_tool_count` fully
removed with no dangling references; the two health endpoints thread the audience
and call `_ensure_den_path` before probing; `renderMcpServers` has no summary dot,
per-server status dot, or per-tool bulb but keeps grouping + toggles + runners +
disabled dimming; dead CSS removed; tests are exact-equality (not weakened) and
would fail on revert; CHANGELOG + NOTES match the code.

No findings.

### Pending user check (manual DoD)

- manual: On the running dashboard, the Health card shows the per-server MCP rows
  (orchestrator two: `mcp: scufris` + `mcp: den`; a sub-agent one: `mcp: agent`)
  and the "MCP tools" dropdown has no coloured circles, only the organized tool
  cards. The backend emission and frontend removal that back this are both
  confirmed by passing tests; the live visual is for the operator to confirm.
