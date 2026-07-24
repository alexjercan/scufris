# Align orchestrator->sub-agent steer path to idle timeout (mcp_server _CHAT_TIMEOUT, opencode client)

- STATUS: OPEN
- PRIORITY: 88
- TAGS: bug,agents,backend

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

- [ ] Trace how `message_agent` -> `_api_call` -> `httpx.request` applies the
      timeout; confirm whether a silent-but-alive sub-agent turn > 120s trips it.
- [ ] Make the orchestrator steer bound idle-based: construct an explicit
      `httpx.Timeout(read=..., connect=short)` rather than a scalar total, so
      only genuine silence past the bound fails. Add/adjust a test in
      tests/test_mcp_server.py driving a slow-but-streaming fake chat response.
- [ ] Inspect `OpencodeClient` (`scufris/opencode_client.py`) timeout wiring;
      ensure `agent_timeout_seconds` is read as a per-read timeout consistent
      with the runner. Adjust + test if it is a total.
- [ ] Full check suite green.

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
