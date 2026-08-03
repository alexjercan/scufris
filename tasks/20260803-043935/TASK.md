# Fix the order-dependent failure in test_orchestrator_chat_uses_server_cwd

- PRIORITY: 55
- TAGS: bug,test
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Story

As a developer trusting a green suite, I want
`tests/test_app.py::test_orchestrator_chat_uses_server_cwd` to pass under every
`pytest-randomly` ordering, so that a red run means a real regression rather
than a shuffle.

## Notes

Observed on `master` at `6e39342`, immediately after landing `20260801-100419`:

- `python -m pytest` failed once with
  `FAILED tests/test_app.py::test_orchestrator_chat_uses_server_cwd - AssertionE...`
- The same test passes in isolation, and the full suite passes with
  `-p no:randomly` (exit 0, 989 passed, 1 skipped) and on repeat runs.

So it is order-dependent, not a regression from that task: the landed diff
touches `_build_telegram_settings_ops` in `scufris/app.py`, nothing on the
server-cwd path.

The seed of the failing run was not captured. Reproduce by looping
`python -m pytest tests/test_app.py -p randomly` over seeds, or run the full
suite with `-p randomly --randomly-seed=<n>` until it reproduces, then find the
leaked state (most likely a process cwd or an app-level singleton another test
mutates).
