# codex/claude sub-agent run errors on >64 KiB app-server line (default readline limit)

- PRIORITY: 0
- TAGS: backlog, bug, agent, codex
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As an operator delegating work to a sub-agent, I want the agent run to survive
a large command output, so that agents can work on big repositories (like
scufris itself) instead of erroring out during orientation.

A codex sub-agent (mode `edit`, backend `codex`) launched against the scufris
project fails: it runs for ~30s doing read-only orientation, then enters
`state: error` before making any edits. The same agent against a small repo
(`first-project`) works fine. The error is NOT about scufris being the
orchestrator's own home or self-modification: the agent dies during read-only
orientation, and the trigger is command-output SIZE, which just happens to be
large in scufris (225 task dirs, ~90 KB `LESSONS.md`, large `AGENTS.md`).

## Root cause

`_stream_app_server` (`scufris/agent.py:477`) launches `codex app-server` with
`asyncio.create_subprocess_exec(...)` at `agent.py:530` and passes NO `limit=`,
so the stdout `StreamReader` uses asyncio's default limit of 65536 bytes. The
JSON-RPC read loop uses `await proc.stdout.readline()` at `agent.py:409` and
`agent.py:626`.

When a single app-server notification line exceeds 64 KiB (e.g. the model runs
`rg -n "System|auth_mode|sandbox_mode|agent|settings|orchestrator" .` over the
whole scufris repo, or dumps a big file / `tatr ls` over 225 tasks), the
aggregated command-output frame is one line larger than 64 KiB. `readline()`
then raises `ValueError: Separator is not found, and chunk exceed the limit`.

`_stream_app_server` only catches `TimeoutError` and `AgentUnavailable`
(`agent.py:661-667`), so the `ValueError` is uncaught. It propagates to the
supervisor's generic `except Exception` (`supervisor.py:291-295`), which sets
`run.error` and marks the run `ERROR`. The orchestrator's `agent_status` only
reports `read_status` fields, not `run.error`, so the failure looked like
"error with no diagnostic message".

Standalone repro of the asyncio behavior (no scufris involved):

```python
proc = await asyncio.create_subprocess_exec(sys.executable, "-c",
    "print('x'*200000)", stdout=asyncio.subprocess.PIPE)
await proc.stdout.readline()
# ValueError: Separator is not found, and chunk exceed the limit
# asyncio.streams._DEFAULT_LIMIT == 65536
```

Evidence from the real failing runs is in
`~/.codex/sessions/2026/07/27/` (orchestrator `019fa315-1ce4`; sub-agents
`019fa315-5fc6`, `019fa316-1d28`): the sub-agent rollouts end abruptly
mid-command right after issuing the large `rg`/ledger reads, the small first
read (a 2 KB `sed`) was recorded fine, and `agent_status` reported
`turns: 1, tool calls: 0, tokens 0/0, state: error`.

## Second occurrence (same bug, claude backend)

`ClaudeBackend.stream` has the identical latent defect: `create_subprocess_exec`
with no `limit=` (`scufris/backends.py:663`) and `proc.stdout.readline()`
(`backends.py:676`). A claude sub-agent on a large repo (or with a large
`stream-json` frame) will fail the same way. Fix both backends.

## Plan (decisions baked in)

- Constant: `STREAM_READ_LIMIT = 8 * 1024 * 1024` defined in `scufris/agent.py`
  (next to the `Stream*` event classes), imported into `scufris/backends.py`
  (which already imports `StreamError` from `.agent`). Both launch sites pass
  `limit=STREAM_READ_LIMIT`, so the intent is documented once and shared.
- Wrapping mechanism:
  - Streaming loops (`agent.py:626`, `backends.py:676`): `try/except ValueError`
    around `readline()` -> `yield StreamError(detail=...)` then stop. This makes
    the over-limit case behave EXACTLY like the existing idle-timeout path
    (`agent.py:665`), which already yields a `StreamError` and ends the run in
    `DONE` state with the error on the bus. Not a regression: it replaces an
    uncaught `ValueError` (bare supervisor `Exception`) with the codebase's
    established terminal-error event.
  - Handshake read (`agent.py:409`, `_appserver_call`, which raises rather than
    yields): `except ValueError` -> `raise AgentUnavailable(<diagnosable>)`,
    reusing the existing `except AgentUnavailable -> StreamError` mapping in
    `_stream_app_server`.
- Optional step (surface `run.error` via `agent_status`): DEFERRED to its own
  task. It is orchestrator-visibility, not the streaming fix, and affects every
  backend `StreamError` equally (the idle-timeout path has the identical gap).
  Out of this task's DoD; file a follow-up rather than widen scope here.

## Steps

- [x] Reproduce FIRST (codex): add a fake app-server body that emits ONE
      `item/agentMessage/delta` notification line > 64 KiB then `turn/completed`,
      and a test that runs it through `_stream_app_server`. Before the fix this
      test raises `ValueError` out of the `async for` (uncaught); it is the
      red repro. `tests/test_agent.py`, real-subprocess fake pattern.
- [x] Raise the limit: define `STREAM_READ_LIMIT = 8 * 1024 * 1024` in
      `agent.py` and pass `limit=STREAM_READ_LIMIT` to `create_subprocess_exec`
      in `_stream_app_server` (`agent.py:530`) and `ClaudeBackend.stream`
      (`backends.py:663`, importing the constant). Turns the repro GREEN: the
      >64 KiB line streams through, delta received, `StreamDone` reached.
- [x] Defense in depth: wrap the `readline()` sites so an over-limit `ValueError`
      becomes a clean `StreamError` / `AgentUnavailable` at `agent.py:409`,
      `agent.py:626`, and `backends.py:676` (see Plan above for the exact
      mechanism per site).
- [x] Test the wrapping (both backends): with `STREAM_READ_LIMIT` monkeypatched
      tiny, an over-limit line yields a `StreamError` event (codex, real fake
      subprocess) and the claude loop yields a `StreamError` when `readline`
      raises `ValueError` (fake stdout) - no uncaught `ValueError` either way.
- [x] File the deferred orchestrator-visibility follow-up as a new backlog task (20260727-140443).

## Definition of Done

- A large single app-server line no longer errors the run
  (test: new large-line streaming test in `tests/test_agent.py` or
  `tests/test_backends.py`).
- Both codex and claude backends pass an explicit `limit=` to
  `create_subprocess_exec` (cmd: `rg -n "create_subprocess_exec" scufris/agent.py scufris/backends.py` shows a `limit=` at each app-server/claude launch).
- An over-limit read yields a `StreamError`, not an uncaught `ValueError`
  (test: assert a `StreamError` event when the limit is artificially tiny).
- Full QA gate green (cmd: `nix flake check`).
- Manual: a codex sub-agent (mode `edit`) delegated against the `scufris`
  project completes orientation and proceeds to real work instead of erroring
  ~30s in (manual: run it from the orchestrator and confirm no early error).

## Notes

- Files: `scufris/agent.py` (`_stream_app_server` at 477, `create_subprocess_exec`
  at 530, `_appserver_call` readline at 409, stream loop readline at 626, except
  clauses at 661-667); `scufris/backends.py` (`ClaudeBackend.stream` 622-694,
  launch 663, readline 676); error handling in `scufris/supervisor.py:291-295`.
- The idle guard (`agent_timeout_seconds`, default 120s) and heartbeat
  (`agent_heartbeat_seconds`, default 600s) are unrelated: the run died ~30s in,
  not on a timeout.
- Related: `tasks/20260724-081811` (spike: auto-retry an agent turn on genuine
  stall / transient app-server failure) - this ValueError currently looks like
  a hard failure that such a retry would also need to classify correctly.
- Bug playbook (AGENTS.md): reproduce with the highest-fidelity harness first,
  then fix, then let the same test pin it.
