# Review: Test isolation - _ensure_den_path leaks SCUFRIS_DEN_PATH across tests

- TASK: 20260727-130139
- BRANCH: fix/den-env-test-isolation

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

The reviewer verified independently: `nix flake check` green (mypy, pytest,
ruff); the leak repro (temporary `.env` with `SCUFRIS_DEN_PATH`, then
`test_app.py` + the once-failing `test_backends` test) green with the fix and
red when the restore logic is reverted; production `_ensure_den_path` is
byte-unchanged (Notes constraint respected); both new tests genuinely fail with
the fix reverted and do not depend on inter-test ordering. In-session
re-derivation: repro and the byte-unchanged production path independently
confirmed here too.

No BLOCKER/MAJOR/MINOR findings. Two NITs, both addressed for clarity:

- [x] R1.1 (NIT) tests/test_env_isolation.py:44 - the `_ensure_den_path(...)`
  call in `test_restore_reinstates_a_pre_existing_scufris_key` is dead: the key
  pre-exists so `setdefault` is a no-op, and restore is proven by the manual
  overwrite. Drop it so the test reads as a pure snapshot/restore-of-preexisting
  check.
  - Response: Removed the dead call and its `type: ignore`; the test now sets the
    baseline, mutates directly, and asserts restore reinstates the baseline.
- [x] R1.2 (NIT) tests/conftest.py:83-90 - note that the snapshot deliberately
  includes the just-set `SCUFRIS_STATE_DIR` and that monkeypatch's own teardown
  is the authority for it. No behavior change needed.
  - Response: Added a one-line comment at the snapshot site recording that
    monkeypatch's teardown is the authority for `SCUFRIS_STATE_DIR`.
