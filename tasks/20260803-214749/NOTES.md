# Notes: Move the host control client into packages/hostctl

Goal in one line: lift the unprivileged client that drives `hostd` - actions,
approvals, the socket client and the NixOS change flow - into `scufris-hostctl`,
along with the two tables it owns.

This is the HARDEST of the three host carves and the only one that is not a pure
move. Its candidate files import six root modules that are not `core` and not
`hostd`, and one of the five named files cannot move at all.

## What changes

Nothing an operator sees. Propose -> preview -> approve -> apply -> audit ->
roll back behaves identically, the `/api/host/actions/*` and `/api/hostconfig/*`
routes serve the same payloads, and the `host_action` / `config_change` rows keep
their columns.

What a MAINTAINER sees:

| Before | After |
|---|---|
| five root modules + `hostconfig/` scattered beside the agent stack | `packages/hostctl`, distribution `scufris-hostctl`, import root `scufris_hostctl` |
| `HostActionRow` / `ConfigChangeRow` in the central `db/models.py` | owned by `hostctl`, registered on `core`'s `Base` |
| the approval flow provable only through the app | `examples/hostctl_approval_flow.py`, offline, gated |

## Surfaces

Moves to `packages/hostctl/src/scufris_hostctl/` (~2050 lines):

| File | Lines | Root imports it must shed or carry |
|---|---|---|
| `host_approvals.py` | 544 | `.host_actions`, `.hostclient`, `.hostd.*`, **`.supervisor` (`RunState`)** |
| `host_actions.py` | 460 | `.db` (`Database`), `.db.models` (`HostActionRow`), `.hostd.*` |
| `hostconfig/changes.py` | 374 | `..db`, `..db.models`, **`..host.run`**, `..hostd.executor` |
| `hostclient.py` | 311 | **`.eventbus`**, **`.supervisor`**, `.hostd.*` |
| `hostconfig/resolve.py` | 228 | **`..host.run`** |
| `hostconfig/service.py` | 176 | **`..config` (`Settings`)**, `..eventbus`, `..hostd.audit` |
| `hostconfig/models.py` | 133 | **`..eventbus`**, **`..supervisor`** |
| `hostconfig/__init__.py` | 99 | internal only |
| `hostconfig/render.py` | 55 | internal only |

Bold = a dependency the epic's graph (`hostctl <- core, hostd`) does not allow.

Does NOT move:

| File | Why |
|---|---|
| `host_approval_bridge.py` (165) | the Steps already say so: it imports `agent_store`, `orchestrator`, `projects` |
| **`host_watch.py` (173)** | the Steps say move it; it cannot. See open question 1 |

Tests to `packages/hostctl/tests/`: none cleanly. `test_host_actions.py` is a
`hostd` test despite its name; `test_nixos_config_change.py` boots the whole app
(`create_app`, `CSRF_HEADER`, `Collector`). See open question 5.

Row classes moving out of `scufris/db/models.py`: `HostActionRow` (models.py:262,
`host_action`) and `ConfigChangeRow` (models.py:298, `config_change`).

Edited: `scufris/api/host.py`, `scufris/api/hostconfig.py`, `scufris/app.py`,
`scufris/db/models.py`, `scufris/db/migrations/env.py` (must import
`scufris_hostctl`'s models so the tables stay on `Base.metadata`),
`scufris/host_approval_bridge.py`, `scufris/host_watch.py`, `scufris/README.md`.

## Data and interfaces

Public facade (new `scufris_hostctl/__init__.py`), assembled from what the app
already imports:

```python
# from host_actions.py
class HostActionStore: ...
class HostActionRecord(BaseModel): ...
# from host_approvals.py
class HostApprovalService: ...
class ConfirmationRequired(Exception): ...
def decision_message(...) -> str: ...
# from hostclient.py
class HostdClient: ...
class HostdError(Exception): ...
class HostdUnavailable(Exception): ...
class HostApplyEvent(BaseModel): ...
HostSupervisor = Supervisor[HostApplyEvent]
HostApplyBus = EventBus[HostApplyEvent]
# from hostconfig/
class ConfigChangeBuilder: ...
class ConfigChangeService: ...
class ConfigChange(BaseModel): ...
def render_change(...) -> str: ...
```

Private, and no sibling may import them under the epic rule:
`scufris_hostctl.models` (the two row classes) and `scufris_hostctl.hostconfig.models`
- note the name collision with the rule's target word; see open question 6.

The generic infrastructure that has to land somewhere before this task can run:

```python
# scufris/eventbus.py - 130 lines, imports NOTHING from scufris
class EventBus(Generic[T]): ...
# scufris/supervisor.py - 450 lines
class Supervisor(Generic[T]): ...   # generic
class RunState(BaseModel): state: RunPhase   # -> scufris/enums.py
AgentSupervisor = Supervisor[StreamEvent]    # <- the ONLY agent coupling, l.419-434
```

`Supervisor` is generic in everything except its three trailing agent lines and
`RunPhase`. Splitting it is a ~15-line edit.

## Sketches

Illustrative only.

```diff
# packages/hostctl/pyproject.toml
+dependencies = [
+  "scufris-core",     # Database, Base, EventBus, Supervisor  (open q. 2)
+  "scufris-hostd",    # protocol, actions, audit, executor    (open q. 4)
+  "scufris-host",     # host.run: Runner, run_command, nix_cli (open q. 3)
+  "pydantic>=2.0.0",
+]
```

```diff
# scufris/supervisor.py  ->  packages/core/src/scufris_core/supervisor.py
-from .agent import StreamError, StreamEvent
-from .enums import RunPhase
 ...
-AgentSupervisor = Supervisor[StreamEvent]
-def _agent_error_event(detail: str) -> StreamEvent: ...
+# the three agent-shaped lines stay in scufris/ (later: packages/agents)
```

```diff
# scufris/hostconfig/service.py
-from ..config import Settings
-        settings: Settings,
+        # narrowed: the two fields it actually reads
+        config_repo: Path | None,
+        config_attr: str | None,
```

## Shape

The privilege trio, after all three carves:

```
   packages/host        read-only, no privilege, no db
        ^
        | run.py: Runner/run_command
        |
   packages/hostd       ROOT, separate process, owns the verbs + audit
        ^
        |  unix socket  (the one real process boundary in the tree)
        |
   packages/hostctl     unprivileged client: build -> preview -> hold ->
        ^               approve -> dispatch -> watch -> record
        |
   scufris/             api/host.py, api/hostconfig.py, host_approval_bridge.py,
                        host_watch.py  (the coupling to agents/projects/chat)
```

What `hostctl` needs from `core` and cannot get from `hostd` or `host`:

```
   core: Database, Base        (host_actions, hostconfig/changes)
         EventBus              (hostclient, hostconfig/models+service)
         Supervisor, RunPhase  (hostclient, host_approvals, hostconfig/models)
```

## Consequences and open questions

Cost: this is the one carve that requires editing code, not just moving it -
`Supervisor` and `EventBus` have to be hoisted, `Settings` has to be narrowed
out of `hostconfig/service`, and one file leaves the task's scope. Bought: the
completed host-agency pillar is parked behind a boundary and can be ignored for
the rest of v0.2.0, which is the entire point of doing it before the rewrite.

Forecloses: nothing structural, but it makes `core` bigger than 20260803-214746's
NOTES scoped it (engine + `Database` + `Base`). See open question 2 - that is a
decision 746 should record, not one this task should discover.

**Open questions for the planner.** 1-4 change what gets written.

1. **`host_watch.py` cannot move, and the Steps say it must.** It imports
   `agent_diagnostics`, `agent_store` (`ORCHESTRATOR_ID`, `AgentStore`), `checks`,
   `config`, `digest`, `health`, `host`, `host_approvals`, `hostclient`,
   `hostd.audit` and `scheduler` - eleven root modules, most of them the agent
   stack that v0.2.0 deletes. Moving it would make `hostctl` import agents and
   projects, i.e. a cycle with the composition root. **Recommendation: drop it
   from the Steps and leave it at the root beside `host_approval_bridge.py`,
   for the same reason the Steps already give for that file.**

2. **`EventBus` and `Supervisor` must land somewhere first.** `hostclient`,
   `host_approvals` and `hostconfig/models` all need them.
   - `eventbus.py` (130 lines) imports nothing from `scufris`. Clean move to `core`.
   - `supervisor.py` (450 lines) is generic except `RunPhase` (enums.py) and its
     last 15 lines (`AgentSupervisor`, `_agent_error_event`, `_agent_error_detail`,
     which import `.agent`). Split: generic half to `core`, agent aliases stay.
   - `RunPhase` then travels to `core` with it. This CONTRADICTS 746's NOTES
     open question 1, which recommends leaving all of `enums.py` at the root -
     defensibly, since `RunPhase` is the supervisor's own phase and not a domain
     enum, but it must be decided in 746 and not here.
   - Alternative: leave both at the root and inject them. Rejected - they are
     TYPES in signatures (`HostSupervisor = Supervisor[HostApplyEvent]`), so
     injection does not remove the import.

3. **`hostctl -> host` is a real edge the epic does not declare.**
   `hostconfig/changes.py:23` and `hostconfig/resolve.py:17` use
   `host.run.Runner`, `run_command`, `nix_cli`, `Outcome`. Combined with
   `hostd -> host` (20260803-214747 open q. 1), the honest graph is
   `core <- host <- hostd <- hostctl <- scufris`. Amend the epic once, covering
   both tasks.

4. **`hostctl` needs more of `hostd` than "the shared protocol types".** The
   Steps say protocol types only; the code imports `hostd.actions` (`ActionKind`,
   `RiskClass`), `hostd.audit` (`Requester`, `AuditRecord`), `hostd.protocol`
   AND `hostd.executor` (`Executor`, `run_action` - `hostconfig/changes.py:24`
   uses the executor to run UNPRIVILEGED `nix build`). All are facade exports, so
   the import rule is satisfied; the Steps' wording is just wrong and should be
   corrected so a reviewer does not read the executor import as a violation.

5. **No test file moves cleanly.** `test_host_actions.py` (imports `hostd` +
   `host.run`) is a hostd test; `test_nixos_config_change.py` and
   `test_host_action_api.py` boot `create_app`. The DoD's
   `python -m pytest packages/hostctl/tests` would run an empty directory.
   Either the planner splits `test_nixos_config_change.py` into a service-level
   half and an app-level half, or the DoD drops that line and rests on the
   example plus the surviving root tests. **Recommend the split** - the service
   half is what proves the move preserved behavior.

6. **`scufris_hostctl.hostconfig.models` collides with the epic's import rule
   word.** The rule forbids importing a sibling's `models` module; here `models`
   holds `ConfigChange` and the bus aliases, which the app DOES need. Resolve by
   re-exporting them from the package facade (the app then imports
   `scufris_hostctl`, never the submodule), and by writing
   `test_no_package_imports_a_sibling_private_module` against the top-level
   `models`/`repo` names only. Worth deciding in 746 where the test is written.

7. **`Settings` in `hostconfig/service.py`.** It reads exactly two fields
   (`host_config_repo`, `host_config_attr`). Narrowing the constructor to two
   parameters is a two-call-site change with no behavior difference - the app is
   the only caller. Cheaper and more honest than making `hostctl` depend on the
   composition root's settings object.

8. **The migration metadata.** Once the two row classes live in `hostctl`,
   `scufris/db/migrations/env.py` must import `scufris_hostctl`'s models before
   reading `Base.metadata`, or autogenerate emits `drop_table` for `host_action`
   and `config_change`. This is the exact failure mode the epic's
   `test_every_package_model_is_registered` exists for, and this task is the
   first one where that test is not vacuous. It is in the DoD already; make sure
   it is written to fail if the `env.py` import is removed.

9. **Ordering against 20260803-214750.** That task squashes the migration
   history to one baseline generated by autogenerate against the surviving
   models. It must run AFTER this one, which the epic's priorities already do
   (p102 then p101). Do not reorder those two.
