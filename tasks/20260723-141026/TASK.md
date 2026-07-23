# Fix operator tool console for HTTP-backed tools (own-port base + off-loop run) + revert pending path

- STATUS: CLOSED
- PRIORITY: 37
- TAGS: bug,agents,backend

## Story

As the operator, I want the "try it" tool console (`POST /api/agent/tools/{name}/run`)
to actually reach THIS dashboard and return a result for HTTP-backed tools (like
`pending_agents`), instead of silently hitting the wrong port or hanging the event
loop.

## Context (the report + diagnosis)

An operator clicked "Run" on `pending_agents` (dashboard on port 7000) and got
`error: 404 from ... {"detail":"Not Found"}`. Root cause, dug out over the report:

1. WRONG BASE. `mcp_server._api_base()` reads `SCUFRIS_API_BASE` from the env,
   defaulting to `http://127.0.0.1:8000`. It is set ONLY for the spawned MCP
   SUBPROCESS (`agent._mcp_overrides`), never for the dashboard's OWN process. So
   the in-process tool runner (`app.run_agent_tool` -> `mcp.call_tool`) defaults to
   :8000 - the operator's SEPARATE stale instance ("dev on 7000, 8000 for shared
   state") - which 404s. `.env`'s `SCUFRIS_API_BASE` would NOT help: pydantic
   `env_file` populates the settings model, not `os.environ`, and `_api_base`
   reads `os.environ` directly.
2. SELF-LOOPBACK HANG. Even pointed at :7000, FastMCP runs a SYNC tool with
   `return fn(...)` ON the event loop (verified: `FuncMetadata.
   call_fn_with_arg_validation`), so `pending_agents`' BLOCKING `httpx` call to its
   OWN server would block the loop until it times out - the loopback request can
   never be served. So the base fix ALONE is insufficient; the tool run must go
   OFF the loop.
3. The earlier route hardening (`/api/agents/pending` -> `/api/pending-agents`,
   task 20260723-120507) was NOT the cause - the report was the base bug. Per the
   operator's call, revert the path to the nicer nested `/api/agents/pending`
   (kept declared before `/api/agents/{id}`, guarded by the org-tag test).

The tool WORKS in the real orchestrator flow (separate subprocess, correct
injected base, no self-loopback); this is specifically the in-process debug
console.

## Steps

- [x] Reverted the poll path: `GET /api/pending-agents` -> `GET /api/agents/pending`
      (declared before `/api/agents/{id}`, ordering-guard docstring restored);
      `pending_agents` tool + `acknowledge` docstring + tests + CHANGELOG reverted;
      `_route_tags` special-case dropped (covered by `startswith("/api/agents")`).
- [x] `_ensure_api_base(settings)` in `app.py` does
      `os.environ.setdefault("SCUFRIS_API_BASE", f"http://127.0.0.1:{settings.port}")`,
      called in `run_server` before `uvicorn.run`.
- [x] `run_agent_tool` runs the tool OFF the event loop via
      `await asyncio.to_thread(lambda: asyncio.run(mcp.call_tool(...)))`; ToolError
      -> 422 mapping preserved. (Added `import asyncio` to app.py.)
- [x] Documented the `SCUFRIS_API_BASE` override in `.env.example`.

## Definition of Done

- `_ensure_api_base` defaults to the dashboard's own port and respects an explicit
  override. (test: `test_ensure_api_base_defaults_and_respects_override`)
- Against a REAL running server, `POST /api/agent/tools/pending_agents/run` reaches
  THIS server and returns the empty-pending result WITHOUT hanging (proves both the
  base and the off-loop run; would hang/timeout without the thread fix).
  (test: `test_tool_console_self_loopback` - boots uvicorn on a free port)
- `GET /api/agents/pending` returns the poll and is not shadowed by
  `/api/agents/{id}`; the `pending_agents` tool calls it.
  (test: `test_pending_agents_and_acknowledge_roundtrip`; cmd: real boot)
- `ruff check .`, `mypy`, `python -m pytest` green from the worktree.
  (cmd: `python -m pytest`)

## Notes

- Reverts the path from 20260723-120507 per the operator's choice; the API-base +
  off-loop fixes are the real resolution of the report.
- Lessons: `nix-devshell-import-resolves-to-cwd-source` (verify the live route by
  booting the worktree source, not the built console script).

## Close record (2026-07-23)

What changed:
- `app.py`: `_ensure_api_base(settings)` (setdefault `SCUFRIS_API_BASE` to
  `http://127.0.0.1:{port}`), called in `run_server`. `run_agent_tool` now runs the
  tool via `await asyncio.to_thread(lambda: asyncio.run(mcp.call_tool(...)))` so a
  blocking self-loopback tool cannot hang the event loop. Reverted the poll route
  to `GET /api/agents/pending` (before `/api/agents/{id}`); dropped the
  `_route_tags` special-case. `import asyncio` added.
- `mcp_server.py`: `pending_agents` tool GETs `/api/agents/pending` again.
- `.env.example`: documented the `SCUFRIS_API_BASE` override.

Evidence: suite 369 passed; ruff + mypy clean. The off-loop fix is pinned by
`test_tool_console_self_loopback`, a REAL uvicorn-socket integration test -
sabotage-verified: reverting to `await mcp.call_tool(...)` makes it fail with
`httpx.ReadTimeout` (the self-loopback hangs the loop). `_ensure_api_base` covered
by a unit test (default + override). END-TO-END real boot on a non-default port
(:17001, no `SCUFRIS_API_BASE`): `POST /api/agent/tools/pending_agents/run` ->
`"no agents are waiting for you"` 200, and `GET /api/agents/pending` -> `[] 200`.

Diagnosis chain (three layers, only the third was truly load-bearing):
1. Reported `404 "no such agent"` -> the in-process tool runner defaulted
   `SCUFRIS_API_BASE` to `:8000` (set only for the spawned subprocess, never the
   dashboard's own process), hitting the operator's SEPARATE stale :8000 instance.
   `.env`'s value would not help: pydantic `env_file` populates the settings model,
   not `os.environ`, and `_api_base` reads `os.environ`.
2. Pointing the base at the own port ALONE would DEADLOCK: FastMCP runs a sync
   tool with `return fn(...)` on the event loop, so the tool's blocking httpx to
   its own server hangs the loop (~15s) - fixed by the off-loop run.
3. The route move (20260723-120507) was never the cause; reverted per the
   operator's call to the nicer nested path.

Difficulties: (1) the `_ensure_api_base` unit test LEAKED `SCUFRIS_API_BASE` into
`os.environ` - its raw `setdefault` is untracked by monkeypatch, so monkeypatch's
restore of a later `setenv` reverted to the leaked value, poisoning 19 mcp_server
respx tests (which assume :8000). Fixed with an explicit snapshot/restore in the
test. (2) `ruff check` (not `format`) flagged the new `import asyncio`'s ordering;
`ruff check --fix` sorted it.

Self-reflection: the report looked like a route bug, then a one-line config
default, and was actually a loop-blocking self-loopback - each layer only visible
after fixing the one above. The real-socket integration test was essential: respx
/ ASGITransport return instantly and would have passed while production hung, the
exact "test green, prod broken" the repo warns about. A function that mutates
`os.environ` directly needs its test to snapshot/restore, not lean on monkeypatch.
</content>
