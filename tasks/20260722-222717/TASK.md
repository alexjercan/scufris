# T1: orchestrator-only scufris MCP scoping (thread is_orchestrator; register server only for orchestrator)

- PRIORITY: 36
- TAGS: spike, telegram, agent, mcp, backend
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

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

- [x] Add `is_orchestrator: bool = False` (keyword-only) to the `AgentBackend`
      Protocol `stream` and to ALL backend implementations - `CodexBackend`,
      `ClaudeBackend`, `MockBackend`, and `OpencodeBackend` (5 signatures total;
      the plan said "three" but the tree has four concrete backends)
      (`scufris/backends.py`).
- [x] Thread the flag from `CodexBackend.stream` into `_stream_app_server`
      (`scufris/agent.py`) and on into `_mcp_overrides`. The other backends accept
      it as a no-op (no MCP wiring).
- [x] In `_mcp_overrides(settings, *, is_orchestrator)`: register the built-in
      `scufris` server block ONLY when `is_orchestrator`. Operator-declared
      `settings.mcp_servers` and the final `approval_policy="never"` still apply to
      EVERY agent (not returned as `[]` - a regular agent may still have operator
      servers and needs the approval policy for an unattended run); only the
      built-in scufris server is orchestrator-scoped.
- [x] At the call site `app.py:_launch_agent_turn`, pass
      `is_orchestrator=(agent.id == ORCHESTRATOR_ID)` into `backend.stream`. The
      only other `stream` caller is the CLI one-shot chat (`cli.py`), which is the
      orchestrator and passes `is_orchestrator=True`. The tool-run path
      (`/api/agent/tools/{name}/run`) does not call `stream`.
- [x] Gate the steering preamble too: `_steer` now takes `is_orchestrator` and
      only steers the orchestrator; server registration and steering stay in
      agreement (both on for the orchestrator, both off otherwise).
- [x] Update the `_stream_app_server` / `_mcp_overrides` / Protocol docstrings to
      state the server is orchestrator-only, and note the claude/mock no-op.
- [x] Tests (`tests/test_agent.py`): `test_mcp_overrides_scopes_scufris_to_orchestrator`
      (unit: False -> no scufris, True -> scufris) and
      `test_stream_app_server_scufris_argv_scoped_to_orchestrator` (harness: real
      spawn, argv carries scufris only for the orchestrator); plus updated the
      existing overrides/steer tests to the scoped API and a new app-level check
      (`test_chat_returns_agent_reply` asserts the landing chat is orchestrator).
      Fixed the test doubles in `test_backends.py` / `test_app.py` to accept the
      new kwarg.

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

## Implementation (close)

What changed: added a keyword-only `is_orchestrator` flag to the backend `stream`
seam (Protocol + Codex/Claude/Mock/Opencode), threaded it Codex -> `_stream_app_server`
-> `_mcp_overrides`/`_steer`. `_mcp_overrides` now emits the built-in `scufris`
server block only for the orchestrator; `_steer` only prepends the tool-steering
preamble for the orchestrator. The two `stream` callers set it: `app.py`'s
`_launch_agent_turn` derives `agent.id == ORCHESTRATOR_ID`, and the CLI one-shot
chat (the main/orchestrator agent) passes True. Operator `mcp_servers` and
`approval_policy="never"` still apply to all agents. CHANGELOG updated (behavior
change).

Design choice: gate at the argv-composition seam (`_mcp_overrides`) rather than
inside the MCP server via an env flag. Both were viable (see SPIKE Q3); composing
the flag out of the codex command line means a regular agent never even spawns the
scufris server, which is cleaner than spawning it and having it self-suppress.

Difficulties: the change rippled to test doubles that fake `_stream_app_server`
and the backend `stream` (in `test_backends.py` and `test_app.py`); they failed
with TypeError until they accepted the new kwarg. The app-chat failures (503 /
RunPhase.ERROR) were the same root cause surfacing through the supervised turn, not
a logic bug. Fixed by widening the fakes' signatures.

Verification: `ruff check`/`ruff format` clean; full `python -m pytest` green (331
tests) incl. the two new DoD tests; my source files add ZERO mypy errors. NOTE the
`nix flake check` DoD item is blocked ONLY by PRE-EXISTING mypy red on master (44
errors, all in test files / other modules; tracked by task 20260720-174021) - the
count is identical with and without this branch, so this task introduces no new
red. Pytest and ruff legs of the gate pass.

Self-reflection: the plan said "three implementations" but the tree has four
backends + the Protocol; grepping `def stream` first caught it before the argv
integration test would have. Writing the argv-level harness test (real spawn,
dump argv) was worth it - it proves the flag reaches the command line, which the
unit test on `_mcp_overrides` alone does not.
