# Retro: B4 per-agent chat endpoint + transcript

- TASK: 20260721-112436
- BRANCH: feature/agent-chat-endpoint
- REVIEW ROUNDS: 1 (out-of-context APPROVE, zero findings)

## What went well

- Spending an out-of-context Explore pass to map the run/supervisor/event-bus/
  backend-stream/transcript plumbing BEFORE writing code paid off: it revealed
  that a chat turn is the run path with a message instead of a goal, so the
  whole task collapsed to (1) extract `_launch_agent_turn` from `run_agent`,
  (2) add a message endpoint that relays the bus inline, (3) add a per-backend
  `read_transcript`. The plan was mechanical because the map was accurate.
- The extraction was a faithful, behavior-preserving refactor (the reviewer
  confirmed `run_agent` is unchanged), so `run` and `chat` now share the
  conflict/session/persist story and cannot drift.
- A pure `parse_claude_transcript` (mirroring the existing `parse_claude_stream`)
  kept the claude reader unit-testable against captured-shape JSONL, and the
  endpoint got a real round-trip test via a seeded claude session on disk.
- Live e2e over a real uvicorn proved the SSE actually streams frames and the
  new routes don't shadow `/{id}` or `/backends` - confidence the buffering
  TestClient can't give.

## What went wrong

- The first 409 (conflict) test DEADLOCKED pytest for >3 minutes (killed it). I
  wrote it as a sync `TestClient.stream(...)` holding the first turn open while
  firing a second request - but BOTH Starlette's TestClient and httpx's
  ASGITransport BUFFER the entire response body before returning, so "hold the
  SSE open without consuming it" is impossible there: the first request never
  returns, and the portal is stuck. Root cause: I reached for the obvious
  same-client approach without recalling the buffering lesson
  (`test-streaming-over-a-real-socket-not-asgitransport`) applies to REQUEST
  concurrency too, not just streaming-timing assertions.
- Fixed by rewriting it async: `httpx.AsyncClient(ASGITransport)` firing
  concurrent requests on one loop - a `create_task` first turn blocked on an
  `asyncio.Event`, a bounded poll of `/status` until running, then a second POST
  that returns 409, then release in a `finally`. Cannot deadlock.

## What to improve next time

- To test "a second request while the first is still in flight" against an ASGI
  app, NEVER hold a buffering-client request open - drive concurrent requests on
  one event loop with `httpx.AsyncClient` (async test) and gate the in-flight
  one on an `asyncio.Event`. Ledgered so the next concurrency test starts there.
- When a landing is blocked by a shared-checkout contention with a parallel
  session, the branch work is safe to leave committed + APPROVED; keep it and
  wait/merge rather than forcing anything - `sprout land` merges into the
  main checkout's CURRENT branch, so landing must wait until it is on master.

## Action items

- [x] Review APPROVE, no follow-ups.
- Next: F4 (chat UI on the detail page), which consumes POST /chat + /transcript
  and reuses the agent-view chat helpers. Completes Milestone 3.
