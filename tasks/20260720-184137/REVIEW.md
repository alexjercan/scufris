# Review: Settings backend - editable tools (per-tool disable + MCP add/remove)

- TASK: 20260720-184137
- BRANCH: feature/settings-editable-tools

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (fresh subagent, ran the full nix suite; in-session
  pass re-ran the suite and adopted both fixes)

No BLOCKER/MAJOR. Full suite green (ruff + mypy + pytest via `python -m pytest`,
162 tests). Enforcement verified genuine: the agent injects
`mcp_servers.scufris.env.SCUFRIS_DISABLED_TOOLS` (value via `json.dumps`, own
argv element - no injection) and `apply_disabled_tools` removes each tool from
the FastMCP registry before serving, so a disabled tool is never advertised or
callable. Global-state test hygiene correct (`restore_tool_registry` restores
the registry in `finally`). Whitelist drift test still green.

- [x] R1.1 (MINOR) app.py:321 + agent.py:398 - id validation used
  `re.match(SERVER_ID_RE, ...)`; Python `$` matches before a trailing newline,
  so `"fs\n"` passed both boundaries and persisted a malformed TOML key.
  - Response: fixed - switched both to `re.fullmatch`, and agent.py now
    compiles the single `SERVER_ID_RE` imported from config.py so the two
    boundaries cannot drift. Added a `"fs\n"` case to the rejection test.
- [x] R1.2 (NIT) test_add_mcp_server_rejects_bad_id only covered a space id.
  - Response: fixed - parametrized it over space, trailing-newline, dot,
    reserved `scufris`, and empty-command cases, so all three rejection
    branches (regex, reserved, empty command) are locked in.

No open `manual:` DoD items.
