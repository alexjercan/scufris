# BC5: end-to-end example + acceptance test (stalled-merge loop self-heals)

- STATUS: OPEN
- PRIORITY: 35
- TAGS: spike,agents,backend

## Story

As a maintainer, I want an end-to-end example and an acceptance test that replay
the stalled-merge scenario, so the bidirectional-comms loop is proven to
self-heal the exact case the spike exists to fix - not just its pieces.

## Context (grounded)

The acceptance scenario (spike): a hello-world sub-agent stops before merging to
master, awaiting confirmation; the orchestrator could not tell "blocked" from
"done" and had no way to poll, so the loop stalled. With BC1-BC4 in place the loop
should close: sub-agent calls `request_input` -> `WAITING` outcome -> orchestrator
woken (BC4) or polls (BC3) -> answers by resuming the sub-agent's session ->
sub-agent proceeds. Per AGENTS.md, features ship with a harness-level test that
drives them the way the app does (async httpx against the FastAPI app, faked
backend), plus a runnable `examples/` script.

Spike: `tasks/20260723-001256/SPIKE.md` (BC5).

## Steps (/plan expands)

- [ ] `examples/` script booting the loop end to end against a faked backend: a
      sub-agent that calls `request_input` awaiting merge confirmation, the
      orchestrator answering by resuming the session, the sub-agent proceeding.
- [ ] An integration test (async httpx + faked backend) replaying the same
      scenario as the acceptance pin.
- [ ] Cover BOTH wake paths: `auto_wake` on (bridge) and off (poll via
      `pending_agents`).
- [ ] Docs sync: README / CHANGELOG note for the new bidirectional-comms surface;
      sweep for stale tool-list references.

## Definition of Done

- The acceptance test drives the full loop against a faked backend and passes:
  request_input -> WAITING -> orchestrator answer-by-resume -> sub-agent proceeds.
  (test: `test_stalled_merge_loop_self_heals`)
- The `examples/` script runs end to end and prints the resolved loop.
  (cmd: `python examples/<script>.py`)
- Both wake paths exercised (bridge + poll). (test: parametrized on `auto_wake`)
- `ruff check .`, `mypy`, `python -m pytest` green from the worktree.
  (cmd: `python -m pytest`)

## Notes

- Depends on BC1-BC4.
- Lessons: harness-first (AGENTS.md), `concurrent-request-test-needs-async-httpx-not-testclient-stream`.
- Spike-seeded (BC5).
