# Fix the flaky test_orchestrator_chat_uses_server_cwd

- PRIORITY: 60
- TAGS: bug,tests
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Story

As the Scufris maintainer, I want `tests/test_app.py::test_orchestrator_chat_uses_server_cwd`
to pass deterministically, so that a red `python -m pytest` means a real
regression rather than a machine that was busy.

Observed on branch `refactor/host-package` (task 20260803-214748) on
2026-08-04: one full-suite run failed the assertion, the same test passed
alone immediately after, and the next full-suite run on the identical tree was
green. Nothing in that branch touches the orchestrator; the test and the code
under it are unchanged from `master`.

The test posts to `/api/agents/orchestrator/chat`, asserts the streamed body
contains `"kind":"done"`, then polls `_wait_state(client, "orchestrator",
"done")` and reads back `session_id`. A streamed turn plus a polling wait is
the classic shape for a load-sensitive flake: the suspicion is `_wait_state`'s
bound, or a race between the stream completing and the agent's state landing
in memory. Which of the three assertions actually failed was not captured -
reproducing it is the first job.

## Notes

- Found while addressing review round 1 of 20260803-214748. Not that branch's
  defect; filed rather than folded in.
- `tests/test_app.py` is on the `check_file_size` ratchet and its split belongs
  to 20260729-103712. Fix the flake in place; do not split the file here.
