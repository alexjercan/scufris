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

**This is the one child that is NOT a pure move.** It requires real surgery
before the move can happen; budget for it.

## Steps

- [ ] Hoist `scufris/eventbus.py` into `core` FIRST. 130 lines, imports nothing
      from `scufris`. `hostclient.py:35,46`, `hostconfig/models.py:15,16` and
      `host_approvals.py:51` all need it, so this is the second consumer that
      justifies it living in `core`. Add it to the `test_core_is_domain_free`
      allowlist with that justification.
- [ ] Split `scufris/supervisor.py` (450 lines). The generic half goes to
      `core`; the agent-shaped region - `supervisor.py:413-450`
      (`AgentSupervisor`, `_agent_error_event`, `_agent_error_detail`, the
      `agent_supervisor()` factory) plus the module-level
      `from .agent import StreamError, StreamEvent` at `:37` - stays at the
      root. This is ~2.5x the "last 15 lines" an early estimate suggested.
- [ ] Move `RunPhase` out of `scufris/enums.py` into `core` with the supervisor.
      It is the supervisor's own phase enum, so it travels with its owner. The
      other nine symbols in `enums.py` stay at the root.
- [ ] Add `pydantic` to `core`'s dependencies. `RunState(BaseModel)`
      (`supervisor.py:72`) needs it, so `core` is no longer sqlalchemy-only.
      Record the change in `tasks/20260803-213242/DECISION.md`.
- [ ] Narrow `Settings` out of `hostconfig/service.py` so `hostctl` does not
      import the root's configuration object.
- [ ] Create `packages/hostctl/` with a `pyproject.toml` depending on
      `scufris-core`, `scufris-host`, `scufris-hostd` and `pydantic`. NOT
      "protocol types only": `hostconfig/changes.py:23` and `resolve.py:17` use
      `host.run`'s `Runner`/`run_command`/`nix_cli`/`Outcome`, and
      `changes.py:24` uses `hostd.executor` to run unprivileged `nix build`.
      All are facade exports, so the import rule is satisfied.
- [ ] Move `scufris/host_actions.py`, `scufris/host_approvals.py`,
      `scufris/hostclient.py` and `scufris/hostconfig/` into
      `packages/hostctl/src/scufris_hostctl/`.
- [ ] Leave `scufris/host_watch.py` at the root. It imports eleven root modules
      - `agent_diagnostics`, `agent_store`, `checks`, `config`, `digest`,
      `health`, `host`, `host_approvals`, `hostclient`, `hostd.audit`,
      `scheduler` - most of them the agent stack v0.2.0 deletes. Moving it would
      make `hostctl` import agents and projects. Same reasoning as
      `host_approval_bridge.py` below.
- [ ] Move the `host_action` (`models.py:262`) and `config_change`
      (`models.py:298`) table definitions with the package. They are `hostctl`'s
      tables; `core` keeps only `Base`.
- [ ] Write `test_every_package_model_is_registered`. This is the first task
      with a package-owned table, so it is the first point the epic's Done Means
      5 can be satisfied. No earlier task creates it.
- [ ] Re-export `hostconfig/models.py`'s `ConfigChange` and bus aliases from the
      package facade. The app needs them, and the import rule forbids reaching
      into a sibling's `models` module - so the facade is how they travel.
- [ ] Leave `scufris/host_approval_bridge.py` in the root for now. It couples
      approvals to the conversation, which does not exist yet - moving it into
      `hostctl` would make `hostctl` import a package that has not been written.
      The epic's open question on whether host approvals are conversation events
      decides where it finally lands.
- [ ] Split `tests/test_nixos_config_change.py` into a service-level half, which
      moves into `packages/hostctl/tests/`, and an app-level half that boots
      `create_app` and stays at the root. No test file moves cleanly otherwise,
      and without the split `pytest packages/hostctl/tests` runs an empty
      directory.
- [ ] Add `examples/hostctl_approval_flow.py`: build an action, preview it
      against a FAKE hostd socket, approve it, dispatch it, and print the audit
      trail. Offline, and added to the gate's opt-in list.
- [ ] Re-point every importer, not only `api/host.py` and `api/hostconfig.py`.
      Also: `scufris/telegram/wiring.py:34,35,40`, `telegram/render.py:36`,
      `telegram/contracts.py:17`, `telegram/approvals.py:25`,
      `telegram/bot.py:25`, `scufris/api/errors.py:15`,
      `scufris/api/agent_runs.py:35`, `scufris/mcp_host_tools/actions.py:81,168`,
      `scufris/app.py`, `scufris/host_watch.py`.

## Definition of Done

- The package imports on its own
  (cmd: `uv run python -c "import scufris_hostctl"`).
- It owns its tables and they are still reachable from the migration metadata
  (test: `test_every_package_model_is_registered`).
- Its own suite passes unmoved in behavior, is not empty, and still runs in the
  canonical gate
  (cmd: `python -m pytest packages/hostctl/tests && python -m pytest --collect-only | rg -q packages/hostctl`).
- `core` grew only by an allowlisted, justified entry
  (test: `test_core_is_domain_free`).
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
