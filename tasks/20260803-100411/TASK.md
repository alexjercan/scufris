# Replace the test suite's fixed-deadline run polling with a bounded wait that fails loudly

- PRIORITY: 0
- TAGS: test,backlog,flake,backend
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Story

As a maintainer, I want the suite's waits on a background run to fail with the
state they actually observed instead of falling through a fixed deadline, so a
loaded machine does not turn a timing lapse into a confusing assertion about a
session id.

## Notes

Observed on 2026-08-03 during the review-fix pass of 20260729-103712:
`tests/test_app.py::test_orchestrator_chat_uses_server_cwd` failed once in a
full-suite run and passed in the two full-suite runs and eight isolated runs
either side of it, with no code change between them.

`tests/test_app.py::_wait_state` polls `/api/agents/{id}/status` 200 times at
10ms and then RETURNS the last state it saw rather than failing. The caller
therefore asserts on a downstream value (`session_id == "mock-session"`) that
is only wrong because the run had not finished inside 2s, so the failure names
the wrong thing.

Not this branch's problem: neither the test nor the helper is touched by
`refactor/extract-remaining-routers` (`git log master..HEAD -- tests/test_app.py`
shows only the env-bridge and factory-reduction commits, and neither diff
mentions `_wait_state`). Recorded as a separate task under
`work/review-feedback.md` section 6.

Direction, not a plan: make the helper raise on timeout with the last polled
state in the message, and look for the other fixed-deadline polls in the suite
(`_wait_state` has at least one sibling shape) rather than fixing this one call
site.
