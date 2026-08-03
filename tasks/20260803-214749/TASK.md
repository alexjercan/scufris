# Move the host control client into packages/hostctl

- PRIORITY: 102
- TAGS: refactor, v0.2.0, architecture, host
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -
- PARENT: 20260803-213242

## Story

As the Scufris maintainer, I want the unprivileged host control client moved
into `packages/hostctl`, so that the completed host-agency pillar is parked
behind one boundary and can be left alone while the rewrite happens next to it.

`hostctl` is the client that DRIVES `hostd`: it builds an action, gets a
preview, holds it for operator approval, dispatches the approved action over the
socket, watches the result, and bridges approval requests out to a channel. Plus
the NixOS configuration change flow with generation rollback.

No behavior changes. This is a move.

## Steps

- [ ] Create `packages/hostctl/` with its own `pyproject.toml` depending on
      `scufris-core` and `scufris-hostd` (for the shared protocol types only).
- [ ] Move `scufris/host_actions.py`, `scufris/host_approvals.py`,
      `scufris/host_watch.py`, `scufris/hostclient.py` and `scufris/hostconfig/`
      into `packages/hostctl/src/scufris_hostctl/`, and their tests to
      `packages/hostctl/tests/`.
- [ ] Move the `host_action` and `config_change` table definitions with the
      package. They are `hostctl`'s tables; `core` keeps only `Base`.
- [ ] Leave `scufris/host_approval_bridge.py` in the root for now. It couples
      approvals to the conversation, which does not exist yet - moving it into
      `hostctl` would make `hostctl` import a package that has not been written.
- [ ] Add `examples/hostctl_approval_flow.py`: build an action, preview it
      against a FAKE hostd socket, approve it, dispatch it, and print the audit
      trail. Offline and gated.
- [ ] Update `scufris/api/host.py` and `scufris/api/hostconfig.py` imports.

## Definition of Done

- The package imports on its own
  (cmd: `uv run python -c "import scufris_hostctl"`).
- It owns its tables and they are still reachable from the migration metadata
  (test: `test_every_package_model_is_registered`).
- Its own suite passes unmoved in behavior
  (cmd: `python -m pytest packages/hostctl/tests`).
- The approval flow is provable without root and without a real socket
  (cmd: `python -m pytest tests/test_examples.py -k hostctl`).
- The privileged path still works end to end on NixOS
  (cmd: `nix build .#checks.x86_64-linux.scufris-vm-test`).

## Notes

- Parent: 20260803-213242.
- Named for its job: it is the client that controls `hostd`. `host` reads and
  needs no privilege; `hostd` is root in another process; `hostctl` is the
  unprivileged client between them.
- This pillar is COMPLETE for the target architecture. Move it, do not improve
  it. Its page gets unlinked later; the code stays.
