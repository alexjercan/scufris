# Review: Backend run-one-tool endpoint + param schema for the 'try it' runner

- TASK: 20260720-134545
- BRANCH: feature/tool-run-endpoint

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Out-of-context reviewer ran the full check suite in the worktree dev shell
(`python -m pytest -q` green, the 3 DoD tests by name green, `mypy scufris` clean
21 files, `ruff check .` / `ruff format --check .` clean, `web/` `npm run lint`
clean + `npm run test` 155 passed) and verified the non-obvious closing-record
claims. In-session pass re-verified the load-bearing consent claim:
`apply_disabled_tools` is called ONLY at scufris/mcp_server.py:339 inside `main()`
(the codex-spawned MCP subprocess), never in `create_app` or any request path - so
the in-process dashboard `mcp` retains every tool and the endpoint's own
`settings.disabled_tools` 403 check is the sole and correct in-process gate. Claim
holds.

Spec/honesty: every ticked step is genuinely done; the "auth/path classification"
non-change is verified (`_route_tags` is OpenAPI tags only, no write-auth gate over
these routes); the confirm step is correctly deferred to the frontend task
20260722-213000; the returned payload is Pydantic JSON (auto-escaped), so
HTML-escaping on render is legitimately the frontend task's job.

Pending manual items: none (the DoD lists only test:/cmd: proofs, all green).

No BLOCKER or MAJOR findings. Remaining items are MINOR/NIT, addressed
opportunistically below since two of them lock the contract the frontend task
consumes.

- [x] R1.1 (MINOR) scufris/app.py (`run_agent_tool`) - a tool that raises at runtime
  (a bug in its own body) is wrapped by FastMCP as `ToolError` and surfaces as 422,
  which is semantically a client error, not a server fault. Latent: scufris tools
  return errors as text rather than raising, and the DoD only requires "never an
  uncontrolled 500" (met). Suggest documenting that all `ToolError`s map to 422 by
  design.
  - Response: fixed - added a docstring note on `run_agent_tool` stating the 422
    mapping is deliberate (FastMCP wraps both arg-validation and in-tool runtime
    errors as `ToolError`; scufris tools do not raise, so 422 is the correct signal
    for the arg-validation case that actually occurs). No behavior change.
- [x] R1.2 (NIT) scufris/app.py (`ToolRunResult`) - no test asserts the `structured`
  field is populated/shaped for any tool, though the frontend runner consumes it.
  - Response: fixed - `test_run_tool_host_stats_returns_result` now asserts
    `body["structured"]` carries the hostname too (host_stats returns a structured
    dict), locking the contract.
- [x] R1.3 (NIT) scufris/app.py (`_tool_parameters`) - the "malformed schema -> []"
  defensive branches are untested; deleting them fails no test.
  - Response: fixed - added `test_tool_parameters_handles_malformed_schema`
    (unit test) exercising a non-dict schema, a missing `properties`, and a
    non-dict property spec.
- [ ] R1.4 (NIT) scufris/app.py (`run_agent_tool`) - `mcp.list_tools()` is walked on
  every run to check existence; negligible for a curated tool set.
  - Response: acknowledged, left as-is. The walk is O(tools) over a small curated
    set and mirrors how `get_agent_tools` already lists them; caching would add
    invalidation complexity for no measurable gain.
