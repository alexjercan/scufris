# Retro: A3 agent run engine

- TASK: 20260720-221942
- BRANCH: feature/agent-run-engine (landed 263a769)
- REVIEW ROUNDS: 1 out-of-context APPROVE (2 MINOR + 1 NIT, all addressed) + in-session round 2

## What went well

- The A0-A2b foundation paid off exactly as designed: the run engine is a thin
  composition (get_backend -> supervisor.start with a stream wrapper + on_complete
  persist). No foundational rework was needed - the seams (cwd, write_enabled,
  read_status, on_complete) all slotted together.
- The reviewer verified every concurrency/lifecycle edge I was worried about
  (unique run id vs re-run, race-free 409, session capture ordering, closed-bus
  replay, delete-during-run) and they all held - the design was sound, not lucky.
- Sweeping for runner-fake stand-ins BEFORE running (the
  protocol-signature lesson) caught the two `sandbox`-typed fakes up front.

## What went wrong

- `run_agent` was first written as a SYNC endpoint. FastAPI runs sync endpoints
  in an AnyIO worker thread, which has no event loop, so `supervisor.start` ->
  `asyncio.create_task`/`get_event_loop` raised "no current event loop". Root
  cause: I did not register that "touches the supervisor" implies "must run on
  the loop thread". Caught at test time, not by inspection.

## What to improve next time

- Any FastAPI endpoint that schedules background work (create_task) or otherwise
  needs the running loop must be `async def`. Treat "calls supervisor.start" as a
  hard signal for an async endpoint - like `/api/chat/stream` already is.

## Action items

- [x] MINORs/NIT addressed (real run-state on launch, best-effort persist
  comment, DoD test names corrected).
- Lesson added: `supervisor-endpoints-must-be-async`.
