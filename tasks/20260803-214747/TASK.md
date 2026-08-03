# Move the root helper into packages/hostd

- PRIORITY: 104
- TAGS: refactor,v0.2.0,architecture,host
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -
- PARENT: 20260803-213242

## Story

As the Scufris maintainer, I want the root helper moved into
`packages/hostd` first and alone, because it is complete, it already talks over
a socket, and it is the only package whose boundary is externally observable -
so it is the cheapest proof that the carve works.

No behavior changes. This is a move.

## Steps

- [ ] Create `packages/hostd/` with its own `pyproject.toml`. It should declare
      almost nothing: the point of moving it first is that a complete component
      has a small dependency list.
- [ ] Move `scufris/hostd/` to `packages/hostd/src/scufris_hostd/` and its tests
      to `packages/hostd/tests/`.
- [ ] Re-point the `scufris-hostd` console script at
      `scufris_hostd.main:main`.
- [ ] Pin `scufris-hostd` to an exact version from the root package.
      `pyproject.toml:29-32` ships it from the same wheel SPECIFICALLY so the
      two halves cannot drift in protocol version; two distributions weaken that
      guarantee to a constraint, so the constraint has to be exact.
- [ ] Add `test_hostd_and_app_report_the_same_protocol_version`, which is what
      replaces the same-wheel guarantee.
- [ ] Update `nix/scufris-hostd.nix` and confirm the NixOS module still resolves
      the binary.
- [ ] Update `scufris/hostd/README.md` and move it with the package.

## Definition of Done

- The helper imports and runs from its own distribution
  (cmd: `uv run python -c "import scufris_hostd"`).
- The two halves cannot drift in protocol version
  (test: `test_hostd_and_app_report_the_same_protocol_version`).
- The existing hostd suite passes unmoved in behavior
  (cmd: `python -m pytest packages/hostd/tests`).
- The privileged helper still builds and activates under NixOS
  (cmd: `nix build .#checks.x86_64-linux.scufris-hostd-vm-test`).

## Notes

- Parent: 20260803-213242.
- `hostd` is COMPLETE for the target architecture. Do not improve it while
  moving it; a behavior change here would be indistinguishable from a carve
  failure, which is the whole reason it goes first.
- It is the only package that legitimately runs as a separate process. Its
  boundary is a unix socket, not an import rule.
