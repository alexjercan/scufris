# Fix the order-dependent failure in test_orchestrator_chat_uses_server_cwd

- PRIORITY: 0
- TAGS: bug,backlog,test
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

## Diagnosis (from 20260801-100441)

The ordering premise above is wrong, and the bug is bigger than one test.

`pytest-randomly` is NOT installed. `pyproject.toml` sets `addopts = "-q"` and
nothing else; collection order is FIXED, and `-p no:randomly` was a no-op that
pytest accepted silently. So "passes with `-p no:randomly`" was never evidence
of an ordering effect, and there is no seed to find. These are TIMING flakes.

The shared root cause is `tests/test_app.py::_wait_state`. It polls status up to
200 times at 10ms and then, on timeout, **returns the last status instead of
failing**:

```python
    for _ in range(tries):
        st = client.get(f"/api/agents/{agent_id}/status").json()
        if st.get("state") == target:
            return st
        time.sleep(0.01)
    return st          # <-- a 2s timeout is indistinguishable from success
```

Every caller then asserts on a field the run had not written yet, so a machine
that missed a 2-second budget reports itself as a wrong VALUE rather than as a
slow run. That is why the failures read as
`assert None == 'mock-session'` and point at the server-cwd or session-persist
path, which are innocent.

Measured while extracting the orchestrator services (same machine, same
session): `tests/test_app.py` alone, 12 runs on that branch and 12 on `master`;
plus 8 full-suite runs each. Two distinct tests failed on the branch -
`test_orchestrator_chat_uses_server_cwd` and
`test_agent_chat_streams_and_persists_session` - both with the
timed-out-then-asserted shape, and both pass 25/25 in isolation.

The fix is to make `_wait_state` raise on timeout (`pytest.fail` naming the
target and the last state seen), then re-measure. A test that times out has to
say so. Whether the 2s budget also needs raising is a separate question that a
loud timeout will answer; the current silence cannot.
