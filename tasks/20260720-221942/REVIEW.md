# Review: A3 agent run engine (launch/status/events + write plumbing)

- TASK: 20260720-221942
- BRANCH: feature/agent-run-engine

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Check suite (reviewer ran it in the worktree): ruff clean, mypy clean (35
files), `python -m pytest` = 245 passed. Verified independently in-session.

The reviewer thoroughly verified the concurrency/lifecycle edges and all held:
unique `agent_id:uuid` run id + `agent_runs` map (re-run of a completed agent
does not collide, status/events retarget to the current run); the 409 check is
race-free (no await between `status()` and `start`); session_id capture is
race-free (set in the stream wrapper before StopAsyncIteration, so `on_complete`
always sees it); events replay a closed/buffered bus and 404 with no run; write
is genuinely default-off (every path defaults read-only, `--sandbox` only on the
first turn); exactly the two `sandbox`-typed runner fakes were updated; no live
writing run; `/api/chat/stream` unchanged; removing `persist` / the merge block
fails a test.

- [x] R1.1 (MINOR) DoD named five tests but only two exist literally; the other
  three assertions are folded into existing tests. Rename to match or note the
  fold.
  - Response: Fixed. The DoD now names the tests that actually exist and states
    what each folds in (status-merge into the reaches-done test; write default
    into the two write tests).
- [x] R1.2 (MINOR) app.py `run_agent` returned `state="running"` unconditionally
  even when the run is `queued` behind the concurrency sem. Return the real
  supervisor state.
  - Response: Fixed. `run_agent` now returns `supervisor.status(run_id).state`
    (usually "queued" until a slot frees); the test asserts `in (queued,
    running)`.
- [x] R1.3 (NIT) app.py `persist` - if the agent is deleted mid-run,
  `mark_finished` raises AgentNotFound (swallowed+logged by the supervisor);
  worth a comment.
  - Response: Fixed. Added a comment noting the delete-during-run case is
    intentionally best-effort.

## Round 2

- VERDICT: APPROVE
- REVIEWER: in-session (two doc/response-shape MINORs + a comment; the only code
  change returns the real run state, covered by the adjusted assertion)

Verification: `run_agent` returns the supervisor's real state; the reaches-done
test asserts `in (queued, running)` at launch and still asserts done + persisted
session_id after. DoD test names now match reality. Suite re-run: ruff + mypy
clean, 245 passed. No new findings.
