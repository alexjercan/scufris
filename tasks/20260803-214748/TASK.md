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

- [ ] Create `packages/host/` with its own `pyproject.toml` declaring `psutil`
      and `pydantic`. NOT `scufris-core`: `scufris/host/*`, `metrics.py` and
      `processes.py` import zero scufris modules and open no database. The epic
      declares `host -> nothing`, and this package is the proof that the rule is
      real.
- [ ] Move `scufris/host/` to `packages/host/src/scufris_host/` and its tests to
      `packages/host/tests/`.
- [ ] Move `scufris/metrics.py` and `scufris/processes.py` with it if they read
      the host and nothing else; leave anything that touches agents or the
      database behind.
- [ ] Update every import in `scufris/`, including `api/host.py:32,33,53,54`,
      `checks.py:46,207` and `host_watch.py:26` - the last two import `.host`
      (`HostInspector`, `Scope`), NOT `.host.run`. `scufris/digest.py` does not
      import `host` at all; leave it alone.
- [ ] Re-point `scufris/telegram/`, which imports `metrics.HostStats` in
      `wiring.py:42`, `render.py:38` and `contracts.py:19`. This makes the real
      `telegram -> host` edge explicit, as the epic's graph now declares.
- [ ] Add `examples/host_report_fixture.py`, an OFFLINE example rendering every
      `Report` from recorded fixtures, and add it to the gate's opt-in list.
      Leave `examples/host_inspect.py` alone - it needs a real NixOS box and
      stays manual, and under the opt-in harness that means simply not listing
      it. Scope the fixture example to the `Runner`-backed reports: `metrics.py`
      calls `psutil` at collector module level and shells out via `subprocess`
      directly, so `HostStats` has no injectable seam and adding one would be
      the behavior change this task forbids.
- [ ] Write `test_stats_endpoint_matches_inspector_output`. It does not exist;
      the nearest coverage is in `tests/test_app.py` and
      `tests/test_route_contract.py`. It is cheap - assert the `/api/stats` body
      equals `Collector().collect()` through a fake - and it is the one
      assertion that makes "Stats still serves the same payload" falsifiable
      across the move.
- [ ] Update the path references naming the moving tree: `README.md`,
      `AGENTS.md:18,19,124,125`, `nix/tests/scufris-vm.nix:66`,
      `web/src/stats-types.ts:103`. Move `scufris/host/README.md` with it.
- [ ] Keep `tests/test_host_nix_store.py` beside `test_host_inspection.py`: it
      does `from test_host_inspection import ok`, a cross-module import that
      only survives because both move together.

## Definition of Done

- The package imports on its own and depends on NO sibling, `core` included
  (cmd: `uv run python -c "import scufris_host"`;
  test: `test_no_package_imports_a_sibling_private_module`).
- Its own suite passes unmoved in behavior AND still runs in the canonical gate
  (cmd: `python -m pytest packages/host/tests && python -m pytest --collect-only | rg -q packages/host`).
- The offline example renders every `Report` from fixtures with no host access
  (cmd: `python -m pytest tests/test_examples.py -k host`).
- Stats still serves the same payload
  (test: `test_stats_endpoint_matches_inspector_output`).

## Notes

- Parent: 20260803-213242.
- This package needs NO privilege. If something here needs root, it belongs in
  `hostctl` or `hostd`.
- Stats is the one page that survives the UI demolition intact. Keep it working
  at every commit.
