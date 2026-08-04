# Fix the flaky agent-fork session-id assertion in test_app

- PRIORITY: 0
- TAGS: bug,backlog,testing,agents
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Story

As a maintainer, I want `test_agent_fork_reverts_single_session` to be
deterministic, so that the `python -m pytest` proof every task carries stops
failing for reasons unrelated to the change under review.

## Steps

- [ ] Reproduce: `tests/test_app.py::test_agent_fork_reverts_single_session`
      fails roughly one full-suite run in three (observed on ca060ff) and
      passes when run alone. The assertion is
      `client.get("/api/agents/builder").json()["session_id"] == "sess-new"`
      seeing `None`, i.e. the seed turn's session was not recorded yet.
- [ ] Find the missing synchronization: `_wait_state(client, "builder", "done")`
      returns on the run state, but the session id is written by a separate
      registry write, so "done" does not imply the id is visible.
- [ ] Make the test wait on the condition it asserts, or make the supervisor
      publish the session id before it publishes `done` - whichever the code
      shows is the real ordering guarantee.
- [ ] Confirm with a repeated full-suite run.

## Definition of Done

- The seed turn's session id is observable whenever the run reports `done`
  (test: `test_agent_fork_reverts_single_session`).
- Ten consecutive full-suite runs are green
  (cmd: `for i in $(seq 10); do python -m pytest -q || exit 1; done`).

## Notes

- Found while establishing the base-branch proof baseline for 20260729-102148.
- Not caused by that branch; it reproduces on master at ca060ff.
