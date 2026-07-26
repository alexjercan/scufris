# Review: journal_* tools fail from the operator tool console

- TASK: 20260727-005013
- BRANCH: fix/journal-console-den-env

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (in-session supplement: A/B re-run - the console test fails
  with the endpoint `_ensure_den_path` call removed, `error: ... not configured`)

The out-of-context reviewer ran the full gate green (`ruff`/`mypy`/`pytest`) and
`nix flake check` green (pure sandbox, `today` absent), and independently verified:
the isolation claim is TRUE (`_AGENT_ROLE_TOOLS = {"request_input"}`; `apply_role`
strips every other tool for a sub-agent, so a subprocess inheriting SCUFRIS_DEN_PATH
is harmless); the A/B (removing the endpoint call reddens
`test_journal_tool_from_console_bridges_den` with the den-not-configured error); `~`
expansion is not double-handled (`_ensure_den_path` writes the raw path,
`mcp_server._journal` expanduser's at use time); and both new tests snapshot/restore
`SCUFRIS_DEN_PATH` (setdefault-leak lesson).

- [ ] R1.1 (NIT) scufris/app.py - `_ensure_den_path` is called in the console ENDPOINT,
  while the task Step 1 first sketched "call it in `run_server`". Deliberate deviation.
  - Response: Correct - moved on purpose. The endpoint is the only in-process journal
    caller, and endpoint placement makes the end-to-end console test work under
    `create_app`/`TestClient` (which never runs `run_server`); the agent path injects
    the den into the subprocess separately, so startup bridging buys nothing. Step 1
    updated to reflect this; recorded in RETRO.

- [ ] R1.2 (NIT) scufris/app.py:_ensure_den_path - per-process `setdefault` pins the den
  for the process lifetime (a later Settings with a different den_path in the same
  process would not propagate).
  - Response: Accepted as-is - Settings is immutable per app instance, and this exactly
    mirrors `_ensure_api_base`'s documented first-value-wins contract. No change.

No BLOCKER/MAJOR/MINOR. No open `manual:` DoD items. APPROVE.
