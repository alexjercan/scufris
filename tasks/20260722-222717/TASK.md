# T1: orchestrator-only scufris MCP scoping (thread is_orchestrator; register server only for orchestrator)

- STATUS: OPEN
- PRIORITY: 36
- TAGS: spike,telegram,agent,mcp,backend

## Goal

Make the scufris MCP server ORCHESTRATOR-ONLY. Thread an `is_orchestrator`
flag from the run call site (`app.py:_launch_agent_turn` knows
`agent.id == ORCHESTRATOR_ID`) through `backend.stream` ->
`_stream_app_server` -> `_mcp_overrides`, and register the scufris MCP server
ONLY for the orchestrator. Regular agents stop receiving it and draw their
tools from their own project `.config` / `.skills` (this is a deliberate
behavior change - today every agent gets the 8 tools).

This is the foundation the Telegram frontend builds on: control tools may only
exist somewhere the orchestrator alone can reach.

## Steps

- [ ] Add `is_orchestrator: bool = False` (keyword-only) to the `AgentBackend`
      Protocol `stream` and to all three implementations - `CodexBackend.stream`,
      `ClaudeBackend.stream`, `MockBackend.stream` (`scufris/backends.py`).
- [ ] Thread the flag from `CodexBackend.stream` into `_stream_app_server`
      (`scufris/agent.py`) and on into `_mcp_overrides`.
- [ ] In `_mcp_overrides(settings, *, is_orchestrator)`: return `[]` (no scufris
      server) when NOT the orchestrator; keep today's registration (plus the
      final `approval_policy="never"`) when it is. Operator-declared
      `settings.mcp_servers` currently ride along here - decide and comment
      whether those stay for all agents or also become orchestrator-only
      (default: keep the built-in `scufris` server orchestrator-only; leave
      operator `mcp_servers` as they are unless the Notes say otherwise).
- [ ] At the call site `app.py:_launch_agent_turn`, pass
      `is_orchestrator=(agent.id == ORCHESTRATOR_ID)` into `backend.stream`.
      Confirm every `backend.stream(` call site is updated (grep) - the tool-run
      path (`/api/agent/tools/{name}/run`) and any test doubles included.
- [ ] Gate the steering preamble too: `_steer` (`scufris/agent.py`) prepends
      `STEERING_PREAMBLE`, which tells the model to prefer the scufris tools -
      meaningless for a regular agent that no longer has them. Thread
      `is_orchestrator` into `_steer` and only steer the orchestrator (the
      non-orchestrator turn gets the bare prompt). Keep the server registration
      and the steering in agreement - both on for the orchestrator, both off
      otherwise.
- [ ] Update the `_stream_app_server` / `_mcp_overrides` docstrings to state the
      server is orchestrator-only, and note the claude/mock no-op.
- [ ] Tests (`tests/test_agent.py` / `tests/test_mcp_server.py` as fits): assert
      `_mcp_overrides(settings, is_orchestrator=False)` emits no `scufris` server
      and `is_orchestrator=True` does; a harness-level assertion that an
      orchestrator turn's codex argv contains the scufris server and a
      regular-agent turn's does not.

## Definition of Done

- `_mcp_overrides` registers the built-in `scufris` MCP server only for the
  orchestrator; a regular agent gets none.
  (test: `` `test_mcp_overrides_scopes_scufris_to_orchestrator` ``)
- Every `backend.stream` call site passes an explicit `is_orchestrator`, and the
  orchestrator call site derives it from `ORCHESTRATOR_ID`.
  (cmd: `grep -rn "backend.stream\|\.stream(" scufris/app.py scufris/supervisor.py`)
- `nix flake check` is green (ruff + mypy + pytest). (cmd: `nix flake check`)

## Notes

- Spike: tasks/20260722-221359/SPIKE.md (Q3, and Context "no orchestrator
  scoping today").
- Foundation task - T2 (control tools) and T3 (prune) build on it.
- Grounding: `scufris/agent.py:_mcp_overrides`/`_server_override`;
  `backends.py` `CodexBackend.stream` / `_stream_app_server` signatures;
  `app.py:_launch_agent_turn` (~1099) passes global `settings` to every
  `backend.stream`; `agent_store.ORCHESTRATOR_ID`.
- Claude backend has no MCP wiring today - keep the claude/mock paths coherent
  (the flag is a no-op there until T2's claude follow-up).
- Test: a regular-agent turn registers NO scufris server; an orchestrator turn
  does.
- spike-seeded; plan into steps with /plan before /work.
