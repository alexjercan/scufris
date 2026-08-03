# Move the root helper into packages/hostd

- PRIORITY: 103
- TAGS: refactor, v0.2.0, architecture, host
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -
- PARENT: 20260803-213242

## Story

As the Scufris maintainer, I want the root helper moved into `packages/hostd`,
because it is complete and its boundary - a unix socket - is externally
observable, so a behavior change here is caught rather than absorbed.

It runs SECOND, after `packages/host` (20260803-214748). It is not import-clean:
six modules import `scufris.host` (`engine.py:33`, `preview.py:26-29`,
`nixos.py:41-43`, `executor.py:25`, `actions/validate.py:18`,
`actions/plans.py:12-13`) and `main.py:17` imports `scufris.logsetup`. Complete
is not the same as standalone.

No behavior changes. This is a move.

## Steps

- [ ] Create `packages/hostd/` with a `pyproject.toml` declaring
      `scufris-host` and `pydantic`. It is NOT dependency-free: see the Story.
- [ ] Move `scufris/hostd/` to `packages/hostd/src/scufris_hostd/` and its tests
      to `packages/hostd/tests/`. `tests/test_host_actions.py` and
      `tests/test_nixos_activation.py` use `FakeRunner`/`CommandResult` from
      `host.run` as doubles - fine, since `hostd` depends on `host`.
- [ ] Re-point the `scufris-hostd` console script at `scufris_hostd.main:main`.
- [ ] Re-point every root importer of `hostd`, not only the six the notes list.
      Also: `scufris/app.py:79,80`, `scufris/api/host.py:51,52`,
      `scufris/api/errors.py:16`, `scufris/api/auth.py:51`,
      `scufris/host_watch.py:29`.
- [ ] Decide `api/errors.py:16` (`from ..hostd.protocol import ErrorCode`). The
      HTTP error mapper depending on the root helper's wire protocol is legal
      under the import rule but worth a deliberate answer, not a silent
      re-point.
- [ ] Re-point `hostclient.py`'s three submodule imports (`.hostd.actions`,
      `.hostd.audit`, `.hostd.protocol`) through the package facade.
      `scufris_hostd/__init__.py` already re-exports all 35 names involved, so
      this costs nothing and makes the boundary greppable.
- [ ] Pin `scufris-hostd` to an exact version from the root package.
      `pyproject.toml:29-32` ships it from the same wheel SPECIFICALLY so the
      two halves cannot drift in protocol version; two distributions weaken that
      guarantee to a constraint, so the constraint has to be exact.
- [ ] Add `test_hostd_and_app_report_the_same_protocol_version`, which is what
      replaces the same-wheel guarantee. `PROTOCOL_VERSION` currently has no
      app-side subject - it appears only in `hostd/protocol.py:34,128` and
      `hostd/__init__.py:39,67` - so the app-side half has to be created, not
      just asserted.
- [ ] Keep `bin/scufris-hostd` on `packages.scufris`. `mkApplication`
      (`flake.nix:113,117`) builds its output from the STRUCTURE of the package
      it is given, so moving the console script to another distribution removes
      that path - which `nix/scufris-hostd.nix:45-50` defaults to and `:147`
      execs. Export a second `mkApplication` and re-point the module default.
      This breaks at BUILD time, so the VM check below is the proof.
- [ ] Add `examples/hostd_socket_roundtrip.py`: drive a verb over a socket
      against a fake privileged backend, offline, and add it to the gate's
      opt-in list.
- [ ] Update the path references that name the moving tree: `README.md:31,135,361`,
      `AGENTS.md:18,19,82,124,125` (`:82` is a live instruction that goes stale),
      `nix/scufris-service.nix:14,138`, `nix/tests/scufris-vm.nix:66`,
      `web/src/host-types.ts:4`. Move `scufris/hostd/README.md` with the package.

## Definition of Done

- The helper imports and runs from its own distribution
  (cmd: `uv run python -c "import scufris_hostd"`).
- The two halves cannot drift in protocol version
  (test: `test_hostd_and_app_report_the_same_protocol_version`).
- The existing hostd suite passes unmoved in behavior AND still runs in the
  canonical gate (cmd: `python -m pytest packages/hostd/tests && python -m pytest --collect-only | rg -q packages/hostd`).
- The console script survives the move to a second distribution
  (cmd: `nix build .#scufris && test -x result/bin/scufris-hostd`).
- The privileged helper still builds and activates under NixOS
  (cmd: `nix build .#checks.x86_64-linux.scufris-hostd-vm-test`).
- The package proves itself offline
  (cmd: `python -m pytest tests/test_examples.py -k hostd`).

## Notes

- Parent: 20260803-213242.
- `hostd` is COMPLETE for the target architecture. Do not improve it while
  moving it; a behavior change here would be indistinguishable from a carve
  failure, which is the whole reason it goes first.
- It is the only package that legitimately runs as a separate process. Its
  boundary is a unix socket, not an import rule.
