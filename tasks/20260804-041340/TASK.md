# Fix the examples the package carve broke

- PRIORITY: 103
- TAGS: bug,v0.2.0,examples
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Story

As the Scufris maintainer, I want `examples/host_agent.py` and
`examples/telegram_approval.py` to run again, so that the runnable examples the
README points at are not silently broken by the package carve.

`6d998c8` (the hostd carve) moved `tests/test_host_actions.py` to
`packages/hostd/tests/test_host_actions.py`. Both examples do their own
`sys.path` setup - `sys.path.insert(0, str(ROOT / "tests"))`
(`examples/host_agent.py:43`) - and import `host_files` / `host_runner` from
that module, so both now die at import with
`ModuleNotFoundError: No module named 'test_host_actions'`.

The pytest suites that import the same module still work, because pytest's own
collection puts `packages/hostd/tests` on the path. Only the examples broke, and
`nix flake check` runs no example (`flake.nix:250-268` is ruff, mypy, pytest,
tatr check, filesize), so nothing caught it.

Found while planning 20260803-214750, which needs `examples/host_agent.py` to
run as a DoD proof.

## Steps

- [ ] Point `examples/host_agent.py:43` and the matching line in
      `examples/telegram_approval.py` at `ROOT / "packages" / "hostd" / "tests"`
      instead of `ROOT / "tests"`, or import the fixtures from wherever the
      hostd package now exposes them.
- [ ] Check the other twelve `examples/*.py` for the same class of stale path
      after the host/hostd/hostctl carves, and fix what is broken.
- [ ] Decide: add an examples smoke check to `flake.nix` `checks` so the next
      carve cannot break them silently, or record why the examples stay
      ungated. If added, keep it cheap - import-and-run with a timeout.

## Definition of Done

- Both named examples run green
  (cmd: `python examples/host_agent.py`;
  cmd: `python examples/telegram_approval.py`).
- Every example runs green
  (cmd: `for f in examples/*.py; do python "$f" || exit 1; done`).
- The examples cannot break silently again, or the reason they stay ungated
  is recorded here (manual: the `checks` attr in `flake.nix` runs the examples,
  or this record's Notes carry the decision not to gate them).

## Notes

- Verify in a FRESHLY SYNCED environment. A stale `.venv` missing
  `_editable_impl_scufris_hostctl.pth` makes five more examples fail with
  `ModuleNotFoundError: No module named 'scufris_hostctl'`. That one is a local
  environment artifact, not a repo defect - `pyproject.toml:53` and `uv.lock`
  both carry the package.
