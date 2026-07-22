# Review: T2 - orchestrator control MCP tools over the local HTTP API

- TASK: 20260722-222722
- BRANCH: feature/orchestrator-control-tools

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Delivers the Goal: five orchestrator-only control tools calling the dashboard's own
HTTP API via a bounded, non-raising `_api_call`, wired through `SCUFRIS_API_BASE` for
the orchestrator server only. Reviewer cross-checked every tool body against the real
FastAPI request models (ProjectCreate/AgentCreate/AgentRunRequest/AgentChatRequest -
all exact), verified the SSE parser matches `_relay_bus_sse`'s frame shapes and the
StreamDone/TextDelta/Error models, confirmed `_api_call` never raises and truncates to
`_MAX_OUTPUT`, and confirmed orchestrator-only scoping. Full suite green; changed
source files add zero mypy errors. respx stubs assert method+path+body and would fail
on revert; error and network paths covered. Two NITs, both addressed in-session:

- [x] R1.1 (NIT) scufris/mcp_server.py - `agent_id` interpolated into the URL path was
  only `.strip()`ed. Not exploitable (ids are slug-confined, body goes via `json=`),
  but the boundary should be explicit.
  - Response: fixed. Added `_clean_id` which rejects an empty id or one containing `/`
    or whitespace; `run_agent` and `message_agent` use it and return an `error:` before
    any HTTP call. Pinned by `test_control_tool_rejects_bad_agent_id`.
- [x] R1.2 (NIT) scufris/mcp_server.py - `import json` was repeated inside two tool
  functions.
  - Response: fixed. Hoisted `import json` to module top and removed the inner imports.

What I re-verified in-session: independently confirmed the tool request bodies match
the app.py request models (field names + required-ness) and that the scufris server is
registered only under `if is_orchestrator` in `_mcp_overrides`.

Open `manual:` DoD items: none.
