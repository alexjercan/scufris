# Bug: agent turn killed at 120s while actively streaming (idle-guard fix)

- STATUS: OPEN
- PRIORITY: 90
- TAGS: bug,codex

## Story

As anyone driving a Scufris agent (a direct chat, or the orchestrator spawning
a sub-agent to implement something), I want a turn that keeps producing output
to run to completion, so that a long-but-progressing turn is not killed at 120s
with `app-server timed out after 120.0s` and reported as "done".

Root cause: `_stream_app_server` (`scufris/agent.py`) computes a single
`deadline = loop.time() + settings.agent_timeout_seconds` (120s) and enforces
it over the WHOLE turn - `initialize`, `thread/start|resume`, `turn/start`, and
every streamed line. Once wall-clock passes 120s the loop `proc.kill()`s and
yields `StreamError("app-server timed out after 120.0s")`, even mid-stream. This
contradicts the ADR-001 supervisor (`scufris/supervisor.py`), which removed the
wall-clock request timeout in favour of a no-output stall guard
(`agent_heartbeat_seconds=600`, `budget_seconds=None`); the leftover runner
deadline defeats it. The reported message string is ours, not codex's.

Decision (see DECISION.md): repurpose `agent_timeout_seconds` as a runner IDLE
(inter-line) timeout, reset on every readline that returns data. A turn times
out only after that many seconds of SILENCE, never for total length. No
auto-retry this run (deferred follow-up).

## Steps

- [ ] Reproduce FIRST: add `test_stream_app_server_slow_but_streaming_completes`
      - fake app-server streams a delta, sleeps < idle, repeats, total > idle;
      with a small `agent_timeout_seconds` (e.g. 0.3s) assert current code yields
      a timeout StreamError (red), and the fix makes it complete with all deltas.
- [ ] Add `test_stream_app_server_idle_stall_times_out` - fake goes silent >
      idle after setup; assert a timeout StreamError is still produced (stall
      guard preserved). Keep both tests sub-second (no real 120s wait).
- [ ] Convert `_stream_app_server`: replace the single shared `deadline` with a
      per-read idle timeout. The streaming loop's `readline` uses
      `timeout=agent_timeout_seconds` directly; the timeout branch fires when a
      readline times out, not when a cumulative wall-clock passes.
- [ ] Apply the same idle semantics to `_appserver_call` (setup handshake reads)
      so a hung `initialize`/`thread/start` is still bounded, per-read not
      cumulative. Drop the now-unused `deadline` plumbing.
- [ ] Update `config.py` `agent_timeout_seconds` docstring: it is now a
      no-output IDLE guard (max silence between app-server lines), NOT a per-turn
      wall-clock; note it complements the coarser supervisor
      `agent_heartbeat_seconds`.
- [ ] Write DECISION.md capturing the idle-guard choice + no-retry-this-run.
- [ ] Run the full check suite (`nix flake check`) green.

## Definition of Done

- A turn that streams across the old wall-clock boundary completes with all
  events. (test: `test_stream_app_server_slow_but_streaming_completes`)
- A turn silent longer than the idle bound still times out.
  (test: `test_stream_app_server_idle_stall_times_out`)
- Existing app-server tests still pass unchanged.
  (cmd: `uv run pytest tests/test_agent.py`)
- `agent_timeout_seconds` docstring documents idle semantics.
  (manual: user reads the diff)
- Full check suite green. (cmd: `nix flake check`)

## Notes

- File pointers: `scufris/agent.py` `_appserver_call` (~361), `_stream_app_server`
  (~386, deadline at ~431, setup calls ~438-508, stream loop ~511-537, except
  ~553). `scufris/config.py:165`. Harness: `tests/test_agent.py` fakes
  (`_FAKE_APPSERVER`, `_write_fake_appserver`) ~300-470.
- The opencode client + orchestrator steer caps are a SEPARATE task (task-2);
  this task is the codex runner only.
- Depends on: none. Blocks: nothing (task-2 is independent code).
