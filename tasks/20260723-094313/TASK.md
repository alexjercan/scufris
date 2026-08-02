# BC4: orchestrator wake bridge (config-gated; defer+batch on 409; no ORCHESTRATOR_ID hold)

- PRIORITY: 36
- TAGS: spike, agents, backend
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As the operator, I want the dashboard to WAKE the orchestrator when a sub-agent
signals it needs input, so a stalled loop self-heals without me driving the
orchestrator by hand - the exact case the spike exists to fix.

## Context (grounded)

The orchestrator is turn-based and unpushable: `_launch_agent_turn`
(`app.py:1074-1145`) is the ONLY way to grant a turn, it raises 409 if a run for
that agent is already queued/running (`app.py:1088-1093`), and it reserves
`serialize_key=agent.id` internally (`supervisor.py:145-179`). Wrapping a launch
inside `supervisor.serialized(ORCHESTRATOR_ID)` self-deadlocks (`app.py:1509-1510`;
lesson `nested-serialized-key-self-deadlock`). So the waker must GRANT a turn via
`_launch_agent_turn`, must NOT already hold `ORCHESTRATOR_ID`, and must cope with
the 409 by DEFERRING the wake (its own pending-wake queue), never dropping it.
FastAPI endpoints that schedule supervisor work must be `async def` (lesson
`supervisor-endpoints-must-be-async`).

Spike: `tasks/20260723-001256/SPIKE.md` (BC4).

## Steps (/plan expands)

- [x] `WakeBridge` (`scufris/wake.py`): on a run completion it enqueues a
      sub-agent with an unacknowledged `WAITING`/`ERROR` outcome (BC1/BC2) and
      drains - granting the orchestrator a turn via an injected `launch`
      (`_launch_agent_turn(orchestrator, wake_prompt)`) with the id(s) + question.
- [x] Pending-wake map ABSORBS the 409 (launch returns False -> stays pending)
      and BATCHES: several completions while the orchestrator is busy fold into
      ONE wake turn, drained when it goes idle (any completion - including the
      orchestrator's OWN - drains). Never dropped.
- [x] The bridge fires from the `on_complete` callback, which runs in the
      supervisor `finally` AFTER the finishing run released its serialize key, so a
      wake `_launch_agent_turn(orchestrator)` never holds `ORCHESTRATOR_ID`
      (no self-deadlock).
- [x] `auto_wake` config key (`config.py`, `SCUFRIS_AUTO_WAKE`, default off,
      `.env.example` doc); off -> `on_run_complete` is a no-op (poll via BC3).

## Definition of Done

- Defer + batch + 409-absorb: while the orchestrator is busy, WAITING sub-agents
  are deferred; when it goes idle its completion drains them as ONE batched wake
  turn; a launch that loses the 409 race keeps the wake pending.
  (test: `test_wake_bridge_defers_and_batches`, `test_launch_409_keeps_pending`)
- END-TO-END: with `auto_wake` ON, a sub-agent whose in-flight run ends with a
  WAITING outcome (real `request_input`) grants the orchestrator a turn carrying
  the question. (test: `test_auto_wake_launches_orchestrator_on_subagent_waiting`
  - async httpx, blocked backend; sabotage-verified against the wiring)
- With `auto_wake` off, no wake turn is launched (polling-only mode).
  (test: `test_auto_wake_off_no_launch` (unit),
  `test_auto_wake_off_does_not_launch_orchestrator` (integration))
- `ruff check .`, `mypy`, `python -m pytest` green from the worktree.
  (cmd: `python -m pytest`)

## Notes

- Depends on BC1 + BC2. Composes with BC3 (poll fallback).
- Lessons: `nested-serialized-key-self-deadlock`, `supervisor-endpoints-must-be-async`,
  `concurrent-request-test-needs-async-httpx-not-testclient-stream` (test the 409
  path with `httpx.AsyncClient(ASGITransport)`, not TestClient).
- SAFETY: a wake grants the orchestrator (now `auto` mode by default,
  `tasks/20260723-001243`) an unattended turn - default `auto_wake` OFF.
- Spike-seeded (BC4).

## Close record (2026-07-23)

What changed:
- `scufris/wake.py` (new): `WakeBridge` + `wake_prompt`. `on_run_complete(agent_id)`
  enqueues a sub-agent with an unacknowledged WAITING/ERROR outcome, then `_drain`
  launches ONE batched orchestrator turn when the orchestrator is idle; a launch
  that returns False (409 race) keeps the batch pending. Collaborators
  (`is_orchestrator_busy`, `launch`) are injected, so the logic is pure and
  unit-testable.
- `app.py`: constructs the bridge in `create_app` with `_orchestrator_busy` (the
  same queued/running check `_launch_agent_turn` 409s on) and `_wake_launch`
  (grants an orchestrator turn, returns False on 409). The run `persist` callback
  calls `wake_bridge.on_run_complete(agent.id)` - AFTER `mark_finished`, in the
  supervisor `finally` past the run's serialize-key release, so no wake ever holds
  `ORCHESTRATOR_ID`.
- `config.py`: `auto_wake: bool = False` (`SCUFRIS_AUTO_WAKE`); `.env.example` +
  CHANGELOG documented.

Evidence: 6 unit tests (defer/batch, 409-absorb, error wakes, done/acknowledged
skip, auto_wake off, prompt format) + 2 async-httpx integration tests (blocked
backend): the ON test drives a real sub-agent run to a WAITING completion and
asserts the orchestrator is granted a turn carrying the question -
SABOTAGE-VERIFIED (stubbing out the `on_run_complete` call makes it fail
"orchestrator was not woken"); the OFF test asserts no wake. Suite 377 passed
(369 baseline + 8); ruff + mypy clean.

Design: the wake triggers on RUN COMPLETION, not at the `request_input` call -
request_input sets the WAITING outcome mid-turn, the turn then ends, and
`mark_finished` PRESERVES that WAITING (BC2's run-id-keyed preservation), which is
what `on_run_complete` reads. Firing from the completion callback is also what
makes deferred wakes drain: the orchestrator's OWN turn ending is a completion, so
it drains the queue when it goes idle. The bridge holds no lock (single event
loop, no await in `_drain`). Errors (`ERROR` outcome) wake too, with a synthesised
message.

Difficulties: none material. The key correctness fact - `on_complete` fires in
`_execute`'s `finally` AFTER `run.state=DONE` and AFTER `release()` - was
confirmed by reading `supervisor.py` before designing, so the "drain sees the
orchestrator as idle" and "no ORCHESTRATOR_ID held" properties hold by
construction. One process nit: the Write tool twice appended a stray `</content>`
line to a new file; caught by a SyntaxError at collection and stripped.

Self-reflection: splitting the bridge into a pure class with injected collaborators
made the tricky logic (defer/batch/absorb) deterministically unit-testable, and the
one async integration test proved the wiring the units can't. Reading the exact
completion-callback ordering in the supervisor up front turned the two scariest
requirements (self-deadlock, drain-when-idle) into by-construction guarantees.
