# Align orchestrator->sub-agent steer path to idle timeout (mcp_server _CHAT_TIMEOUT, opencode client)

- PRIORITY: 88
- TAGS: bug, agents, backend
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As the orchestrator steering or conversing with a sub-agent, I want a
long-but-progressing sub-agent turn not to be cut at a 120s total bound, so
that the idle-guard fix (task 20260724-011406) holds for ALL agents and not
just the direct-chat path.

Two sibling 120s caps sit outside the codex runner and must move to idle
semantics for consistency:

- `scufris/mcp_server.py:267` `_CHAT_TIMEOUT = 120.0` - the orchestrator's
  `message_agent` MCP tool passes this to `_api_call` (httpx) when it POSTs to
  the sub-agent's `/api/agents/{id}/chat` SSE endpoint and buffers the reply.
  A 120s httpx read bound cuts a sub-agent turn that is silent (a long tool
  call) longer than 120s. Make it an explicit idle (per-read) bound, not a
  total, and align its default with the runner's idle timeout.
- `scufris/backends.py:717` OpenCode client `timeout=agent_timeout_seconds` -
  now that `agent_timeout_seconds` is idle-semantics (per task -011406), confirm
  the OpencodeClient applies it as an httpx READ (per-chunk) timeout, not a
  total; adjust if it is a total.

## Steps

- [x] Trace how `message_agent` -> `_api_call` -> `httpx.request` applies the
      timeout; confirm whether a silent-but-alive sub-agent turn > 120s trips it.
- [x] Make the orchestrator steer bound idle-based: construct an explicit
      `httpx.Timeout(read=..., connect=short)` rather than a scalar total, so
      only genuine silence past the bound fails. Add/adjust a test in
      tests/test_mcp_server.py driving a slow-but-streaming fake chat response.
- [x] Inspect `OpencodeClient` (`scufris/opencode_client.py`) timeout wiring;
      ensure `agent_timeout_seconds` is read as a per-read timeout consistent
      with the runner. Adjust + test if it is a total.
- [x] Full check suite green.

## Definition of Done

- The orchestrator->sub-agent steer path does not fail a slow-but-streaming
  sub-agent turn whose total exceeds the old 120s.
  (test: mcp_server slow-stream test)
- OpenCode client timeout is idle/per-read consistent with the runner, verified
  by test or by reading the client. (test: tests/test_opencode_client.py or manual)
- Full check suite green. (cmd: `nix flake check`)

## Notes

- Depends on: 20260724-011406 (establishes the idle-timeout decision; this task
  extends it to the two sibling caps). Independent code, can be built after -011406
  lands.
- Keep tests sub-second; do not wait real seconds.

## Implementation record

Plan refinement from the trace: both sibling caps are SYNCHRONOUS single
requests, not chunk-streaming reads - `OpencodeClient.send_message` blocks on
one POST the daemon answers only when the turn is done, and `message_agent` ->
`_api_call` uses buffering `httpx.request` over the whole SSE body. So an httpx
READ timeout there is NOT a per-chunk idle bound; for a silent turn it is the
total-turn cap. The plan's "make it per-read idle" would therefore not have
fixed it. The actual fix is to DISABLE the read bound (`read=None`) for the
turn call while keeping connect/write/pool capped, and trust the turn to
self-terminate - the codex runner's idle guard (20260724-011406) and the
supervisor heartbeat (600s) are the real backstops that bound the run.

What changed:
- `scufris/opencode_client.py`: `_turn_timeout = httpx.Timeout(timeout,
  read=None)` built in `__init__`; `_request` gained an optional per-call
  `timeout` override; `send_message` passes `_turn_timeout`. Quick calls
  (health/create_session/get_messages) keep the client-default bounded read.
- `scufris/mcp_server.py`: dropped the `_CHAT_TIMEOUT = 120.0` constant; added
  `read_unbounded` to `_api_call` (builds `httpx.Timeout(timeout, read=None)`);
  `message_agent` passes `read_unbounded=True`. Connect stays bounded by
  `_API_TIMEOUT` (15s).

Tests assert the behavior at the request boundary via respx / httpx request
extensions: the turn/steer request carries `timeout["read"] is None` while a
quick call and connect stay bounded. Both would fail pre-fix (read was 30/120).

Self-reflection: the value of tracing before coding - the plan's stated
approach (per-read idle) was subtly wrong for a synchronous request, and only
reading the actual call shape surfaced that read=None (trust-the-turn) is the
correct model. Recorded as a lesson.

Review round 1 (R1.1): out-of-context review caught that `read=None` needs a
backstop, which the supervised dashboard path has (heartbeat) but the one-shot
`scufris chat` CLI path did not - so opencode-via-CLI could hang forever, a
regression this branch introduced. Fixed by giving `_chat_once` its own
no-output guard (`asyncio.wait_for(anext(), timeout=agent_heartbeat_seconds)`),
mirroring `supervisor._drain`; `test_chat_one_shot_stalled_turn_is_bounded`
pins it. This also uniformly stall-guards the codex CLI path.
