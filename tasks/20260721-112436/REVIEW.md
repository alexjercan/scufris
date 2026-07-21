# Review: B4 per-agent chat endpoint + transcript

- TASK: 20260721-112436
- BRANCH: feature/agent-chat-endpoint

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Full suite ran green in the worktree under a 240s timeout (ruff + mypy 35 files +
pytest, finished in seconds - no hang; `asyncio_mode = auto` confirms the 409
test runs as a real coroutine, not skipped). Zero findings.

Verified by the reviewer:
- `_launch_agent_turn` is a faithful mechanical extraction from `run_agent`
  (only `agent_id`->`agent.id`, `goal`->`prompt`, and returning `bus`); the 409
  check, session-capture closure, `mark_running`, `supervisor.start`, and
  `mark_finished` persist are identical, so run + chat both write back the
  (possibly new) session id. `run_agent`'s external behavior is unchanged.
- The chat endpoint validates 404/422-empty-message/422-missing-project/409 and
  passes the STRIPPED message to the backend; it relays the bus inline via
  `_relay_bus_sse` with the same frame shape + headers as `/events`.
- `parse_claude_transcript`: string user turns kept, list tool_result turns
  skipped, assistant text concatenated, tools/usage carried (int-guarded), empty
  assistant frames dropped, limit applied; no None/missing-field crash. Codex
  delegates to sessions.read_transcript; mock returns [].
- Route ordering is safe: `/backends` before `/{agent_id}`; `/chat` + `/transcript`
  are deeper segments that cannot shadow or be shadowed.
- The 409 test genuinely exercises the conflict (bounded 200-iter poll to
  running, then a concurrent second POST -> 409, `finally: release.set()`); the
  async httpx rewrite cannot deadlock. No tests weakened.

No BLOCKER/MAJOR/MINOR/NIT issues. APPROVE.
