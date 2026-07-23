# Review: tool console reaches its own server; revert pending path

- TASK: 20260723-141026
- BRANCH: fix/tool-console-loopback

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (round-1 findings), in-session pass verified and recorded

The out-of-context reviewer ran the full suite (369 passed, ruff + mypy clean),
independently SABOTAGE-verified the load-bearing test (reverted the off-loop fix
-> `test_tool_console_self_loopback` fails with `httpx.ReadTimeout`, then
restored), confirmed the env leak is gone (ran the 19 `:8000` respx tests +
`test_ensure_api_base_...` together -> all pass), confirmed the path revert is
complete (zero stray `/api/pending-agents`) and the route is declared before
`/api/agents/{id}`, and confirmed the threading is sound (no async tools, no
`get_context` usage, `asyncio.run` re-raises so ToolError->422 holds). In session
I had boot-verified the end-to-end scenario on a non-default port. No
BLOCKER/MAJOR/MINOR.

- [ ] R1.1 (NIT) tests/test_app.py `test_tool_console_self_loopback` - the
  `_free_port()` bind-close-rebind has a tiny TOCTOU window; standard and
  low-risk.
  - Response: left as-is (the reviewer agreed it is acceptable). A rare port race
    is the accepted cost of a real-socket test.
- [ ] R1.2 (NIT) scufris/app.py `run_agent_tool` - `to_thread(lambda:
  asyncio.run(...))` spins a fresh loop per run and surfaces a Py3.14
  `asyncio.iscoroutinefunction` DeprecationWarning from `asyncio.threads` (not our
  code).
  - Response: left as-is - it is the simplest shape that avoids the deadlock; the
    warning is from the stdlib, not this code.

No open `manual:` DoD items (all proofs are `test:`/`cmd:`).
</content>
