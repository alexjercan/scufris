# Fix the flaky agent-run session persistence test

- PRIORITY: 40
- TAGS: bug,test,flake
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Story

As the maintainer, I want `nix flake check` to mean "the suite is green", so
that a red run is a reason to stop rather than a reason to re-run it.

## Notes

Seen on 2026-08-04 during the review-fix pass for 20260804-115322, on a branch
that touches only `examples/chat_conversation.py`, `tests/test_examples.py` and
task records - `tests/test_app.py` is not in its diff at all.

    tests/test_app.py:2604 test_agent_run_reaches_done_and_persists_session
    AssertionError: assert None == "mock-session"

The test polls `_wait_state(client, "builder", "done")` and then reads
`/api/agents/builder` expecting the mock run's session id to be persisted.
Reaching `done` and having written `session_id` are two different moments, so
the wait is on the wrong condition: the run reports done while the persistence
that follows it has not landed.

Test-order dependent rather than environment dependent: the suite runs under
`pytest-randomly`, and both sightings are of a run whose order differed, not of
a sandbox. `test_agent_run_reaches_done_and_persists_session` was seen under
`nix flake check`; `test_agent_fork_reverts_single_session` was seen during
round 2 under a plain `nix develop --command python -m pytest`, outside any
sandbox. Neither reproduces every run - the same tree passed the immediately
following invocation - and both are green when the file runs in isolation and
under `-p no:randomly`, which is the order dependence rather than an alibi for
it. Reproduce with a recorded `-p randomly --randomly-seed=<seed>` before
fixing.

Fix the wait rather than sleeping: poll for the persisted `session_id`, or make
the state transition to `done` happen after the write it implies. Expect the
same shape in the fork test.
