# Move read-only host inspection into packages/host

- PRIORITY: 104
- TAGS: refactor, v0.2.0, architecture, host
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -
- PARENT: 20260803-213242

## Story

As the Scufris maintainer, I want read-only host inspection moved into
`packages/host`, so that the one product surface surviving the rewrite
untouched - Stats - sits behind a package that needs no privilege and imports no
sibling but `core`.

No behavior changes. This is a move.

## Steps

- [ ] Create `packages/host/` with its own `pyproject.toml` declaring only
      `psutil` and `scufris-core`.
- [ ] Move `scufris/host/` to `packages/host/src/scufris_host/` and its tests to
      `packages/host/tests/`.
- [ ] Move `scufris/metrics.py` and `scufris/processes.py` with it if they read
      the host and nothing else; leave anything that touches agents or the
      database behind.
- [ ] Update every import in `scufris/` and in `scufris/api/host.py`.
- [ ] Rework `examples/host_inspect.py` into `examples/host_inspect.py` plus a
      new OFFLINE `examples/host_report_fixture.py` that renders every report
      from recorded fixtures. The first needs a real NixOS box and stays manual;
      the second is gated.
- [ ] Mark the host-touching examples as manual in `tests/test_examples.py` so
      the gate runs only what can run anywhere.
- [ ] Move `scufris/host/README.md` with the package.

## Definition of Done

- The package imports on its own and depends only on `core`
  (cmd: `uv run python -c "import scufris_host"`).
- Its own suite passes unmoved in behavior
  (cmd: `python -m pytest packages/host/tests`).
- The offline example renders every report from fixtures with no host access
  (cmd: `python -m pytest tests/test_examples.py -k host`).
- Stats still serves the same payload
  (test: `test_stats_endpoint_matches_inspector_output`).

## Notes

- Parent: 20260803-213242.
- This package needs NO privilege. If something here needs root, it belongs in
  `hostctl` or `hostd`.
- Stats is the one page that survives the UI demolition intact. Keep it working
  at every commit.
