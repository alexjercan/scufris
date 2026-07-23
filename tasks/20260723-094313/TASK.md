# BC4: orchestrator wake bridge (config-gated; defer+batch on 409; no ORCHESTRATOR_ID hold)

- STATUS: OPEN
- PRIORITY: 36
- TAGS: spike,agents,backend

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

- [ ] A config-gated in-app watcher (`auto_wake`, default off per SPIKE) that, on
      a needs-input/error outcome (BC1/BC2), grants the orchestrator a turn via
      `_launch_agent_turn(orchestrator, injected_prompt)` with the sub-agent id +
      question in the prompt.
- [ ] A pending-wake queue that ABSORBS the 409 (orchestrator mid-turn) and
      BATCHES concurrent completions into one "these agents need you: [...]" turn;
      drains when the orchestrator goes idle. A wake is never dropped.
- [ ] The watcher never holds `ORCHESTRATOR_ID` when it launches (avoid the
      self-deadlock).
- [ ] `auto_wake` config key (pydantic-settings, `.env.example` doc); when off,
      the orchestrator falls back to polling (`pending_agents`, BC3).

## Definition of Done

- With `auto_wake` on and the orchestrator mid-turn, a sub-agent `request_input`
  enqueues EXACTLY ONE orchestrator turn carrying the question once the
  orchestrator goes idle (the 409 is absorbed, not dropped).
  (test: `test_wake_bridge_defers_and_batches` - async httpx, two concurrent runs)
- With `auto_wake` off, no wake turn is launched (polling-only mode).
  (test: `test_auto_wake_off_no_launch`)
- `ruff check .`, `mypy` touched files, `python -m pytest` green from the
  worktree. (cmd: `python -m pytest`)

## Notes

- Depends on BC1 + BC2. Composes with BC3 (poll fallback).
- Lessons: `nested-serialized-key-self-deadlock`, `supervisor-endpoints-must-be-async`,
  `concurrent-request-test-needs-async-httpx-not-testclient-stream` (test the 409
  path with `httpx.AsyncClient(ASGITransport)`, not TestClient).
- SAFETY: a wake grants the orchestrator (now `auto` mode by default,
  `tasks/20260723-001243`) an unattended turn - default `auto_wake` OFF.
- Spike-seeded (BC4).
