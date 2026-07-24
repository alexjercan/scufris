# Goal: agent turns never time out while actively streaming (idle-based guard)

- DATE: 20260724
- UMBRELLA TASK: 20260724-081616
- LANDING SCOPE: squash-merge each task to local `master`; do NOT push (user's call).

## Goal

Today any agent turn that runs longer than 120s wall-clock is killed
mid-stream with `app-server timed out after 120.0s`, even while it is actively
producing tokens. The message is emitted by our own runner
(`scufris/agent.py`), where `_stream_app_server` sets a single per-turn
`deadline = now + agent_timeout_seconds` (120s) covering setup + the entire
stream. This defeats the ADR-001 supervisor design (`scufris/supervisor.py`),
which deliberately removed the wall-clock request timeout in favour of a
no-output stall guard (`agent_heartbeat_seconds`, `budget_seconds=None`).

This run makes the runner's timeout an IDLE (inter-line) guard: a turn times
out only after N seconds of SILENCE from the app-server, never for total
length. A turn that keeps streaming - a long conversation turn, or a spawned
sub-agent implementing something for minutes - runs to completion. A genuinely
hung app-server (no output) is still cut, preserving the stall guard. The same
idle semantics are applied to the sibling 120s caps on the path an orchestrator
uses to steer/converse with a sub-agent, so the fix holds for ALL agents, not
just the direct-chat path. Auto-retry on a genuine stall is explicitly OUT of
scope for this run and is filed as a follow-up.

## Done means

1. A turn that streams output across the old wall-clock boundary (total time >
   `agent_timeout_seconds`, but never silent longer than it) completes with all
   its events, instead of a timeout StreamError.
   (test: `test_stream_app_server_slow_but_streaming_completes` in tests/test_agent.py)
2. A turn that goes SILENT longer than `agent_timeout_seconds` still yields a
   timeout StreamError (stall guard preserved).
   (test: `test_stream_app_server_idle_stall_times_out` in tests/test_agent.py)
3. The orchestrator->sub-agent steer/converse path no longer cuts a long-but-
   progressing sub-agent turn at a 120s total bound (idle-based instead).
   (test: mcp_server / message_agent test in tests/test_mcp_server.py)
4. `config.py`'s `agent_timeout_seconds` docstring reflects the new idle
   semantics and its relationship to the supervisor heartbeat.
   (manual: user reads the docstring / diff)
5. Auto-retry is tracked as its own OPEN follow-up task, not silently dropped.
   (cmd: `tatr show <retry-task-id>`)

Overall: the full check suite passes (`nix flake check`), and no reproduction
test relies on a real multi-second wall-clock wait.

## Tasks

Updated as tasks land (one line per land).

- [ ] 20260724-011406 (p90, scufris) Core fix: runner idle-timeout in _stream_app_server + config docstring
- [ ] 20260724-081804 (p88, scufris) Align orchestrator->sub-agent steer path (mcp_server _CHAT_TIMEOUT / opencode client) to idle semantics
- retry follow-up tracked OUTSIDE this umbrella (deferred by user): 20260724-081811 (spike)

## Decisions (load-bearing, architectural)

- 20260724-011406 DECISION.md: repurpose `agent_timeout_seconds` as a runner
  IDLE timeout (reset per streamed line), not a per-turn wall-clock; no
  auto-retry this run. (ACCEPTED)

## Manual acceptance (batched for the user at Finish)

- (pending) 20260724-011406: read the `agent_timeout_seconds` docstring diff and
  confirm the idle semantics read correctly.
