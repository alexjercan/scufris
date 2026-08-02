# journal_* tools fail from the operator tool console (den env not bridged in-process)

- PRIORITY: 50
- TAGS: bug, mcp, journal
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

The journal_* MCP tools work from an agent turn but FAIL from the operator tool
console ("RUN" button), returning "error: the-den journal is not configured (set
SCUFRIS_DEN_PATH)". Make them work from the console too.

## Root cause

`POST /api/agent/tools/{name}/run` (app.py) runs the tool IN-PROCESS via
`mcp.call_tool`, so `mcp_server._den_path()` reads the DASHBOARD process
`os.environ` - which has no `SCUFRIS_DEN_PATH`. That var is only injected into the
AGENT's MCP subprocess (agent.scufris_mcp_server); in dev the value lives in
`settings.den_path`/`.env`, which pydantic reads into Settings but never exports to
`os.environ`. The dashboard already bridges the analogous `SCUFRIS_API_BASE` for the
same in-process console path via `_ensure_api_base()` (app.py:2083).

## Steps

- [x] Add `_ensure_den_path(settings)` mirroring `_ensure_api_base`:
      `os.environ.setdefault("SCUFRIS_DEN_PATH", str(settings.den_path))` when
      `settings.den_path is not None`. Call it in the console ENDPOINT
      (`run_agent_tool`) before running the tool - NOT `run_server` as first
      sketched: the endpoint is the only in-process journal caller, and endpoint
      placement makes the end-to-end console test work under create_app/TestClient
      (which never runs run_server). (review R1.1)
- [x] Test: with `settings.den_path` set and `SCUFRIS_DEN_PATH` absent from the env,
      the bridge populates it; an explicit env value still wins (setdefault); unset
      den leaves it absent. Snapshot/restore the env key (setdefault leaks past
      monkeypatch - see ledger).
- [x] Verify the console path end to end: `journal_show` via
      `/api/agent/tools/journal_show/run` returns the day (with a den configured),
      not the "not configured" error.

## Notes

- Isolation is unaffected: sub-agents cannot call journal_* (apply_role keeps only
  request_input for the agent role), so a subprocess inheriting SCUFRIS_DEN_PATH is
  moot. Confirm in the retro.
- Surfaced by the operator 2026-07-27 after 20260726-225845 wired the deployed env.
