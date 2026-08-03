# Notes: Move read-only host inspection into packages/host

Goal in one line: lift `scufris/host/` - plus `metrics.py` and `processes.py`,
which are the same concern under different names - into `scufris-host`, a package
that needs no privilege, no database and (as it turns out) no `core` either.

This is the cleanest package in the epic: `scufris/host/` imports NOTHING from
`scufris`. It is the only carve that is a pure `git mv` plus import re-pointing.

## What changes

Nothing an operator sees. `/api/stats`, `/api/processes`, `/api/host/*` and the
Stats page serve identical payloads.

What a MAINTAINER sees:

| Before | After |
|---|---|
| `scufris/host/` + `metrics.py` + `processes.py` scattered in the root | `packages/host`, distribution `scufris-host`, import root `scufris_host` |
| `examples/host_inspect.py` needs a real NixOS box, ungated | it stays manual, and a NEW `examples/host_report_fixture.py` is gated and offline |
| "no privilege needed here" is a claim | the package declares `psutil` + `pydantic` and nothing else |

## Surfaces

Moves to `packages/host/src/scufris_host/` (2900 lines):

| File | Lines | Why |
|---|---|---|
| `host/render.py` | 434 | report rendering |
| `host/packages.py` | 374 | nix store, generations, profiles |
| `host/storage.py` | 372 | filesystems, generations, reclaimable space |
| `host/network.py` | 354 | interfaces, listening sockets, firewall |
| `host/thermal.py` | 312 | temperatures and throttle counters |
| `host/__init__.py` | 310 | `HostInspector`, `Scope`, ~45 exported names |
| `host/units.py` | 282 | systemd unit status |
| `host/journal.py` | 224 | journal reads, bounded |
| `host/run.py` | 207 | `Runner`, `run_command`, `CommandResult`, `Outcome`, `nix_cli`, `FakeRunner` |
| `host/models.py` | 106 | `Availability`, `Report`, `Bounded` |
| `host/overview.py` | 93 | `HostOverviewCache`, TTL |
| `host/README.md` | - | moves with the package |
| `metrics.py` | 386 | `Collector`, `HostStats` - psutil + `platform` only, no scufris import |
| `processes.py` | 126 | `ProcessCollector`, `ProcessList` - psutil only |

Tests to `packages/host/tests/`: `test_host_inspection.py`, `test_host_thermal.py`,
`test_host_nix_store.py`, `test_metrics.py`, `test_processes.py`. Each imports
`scufris.host` / `scufris.metrics` / `scufris.processes` and nothing else.

Stays at the root - reads the host but is not host inspection:

| File | Why it stays |
|---|---|
| `scufris/checks.py` | imports `hostd.actions`; it is the health gate, not an inspector |
| `scufris/digest.py` | composes host + agent state for the daily digest |
| `scufris/scheduler.py` | root-owned scheduling |
| `scufris/host_mcp_server.py`, `mcp_host_tools/` | MCP wiring = composition root |
| `scufris/api/host.py` | the HTTP surface |

Edited:

| File | Why |
|---|---|
| `scufris/api/host.py:32,33,53,54` | 4 import lines |
| `scufris/app.py` | wiring imports |
| `scufris/checks.py`, `digest.py`, `host_watch.py`, `hostconfig/{changes,resolve}.py` | `from .host.run import ...` -> `scufris_host` |
| `scufris/hostd/*` (6 modules) | see open question 1 |
| `scufris/telegram/{contracts,render,wiring}.py` | they import `metrics` - see open question 3 |
| `scufris/mcp_host_tools/inspection.py`, `mcp_server.py` | re-point |
| ~14 root test modules | re-point |
| `examples/host_inspect.py`, `host_action.py`, `nixos_change.py` | re-point |
| `web/src/stats-types.ts` | comment reference to `scufris/metrics.py` only |

## Data and interfaces

No signature changes. The package facade is `scufris/host/__init__.py`'s existing
`__all__` (~45 names: `HostInspector`, `HostOverview`, `Scope`, `Runner`,
`CommandResult`, `Outcome`, `FakeRunner`, `render`, the report models). `metrics`
and `processes` join it - either re-exported from `scufris_host` or kept as
`scufris_host.metrics` / `scufris_host.processes` submodules.

Recommendation: submodules, re-exported by name from the facade only where an
existing importer already used the short path. `HostStats` and `ProcessList` are
API response models; making them facade names is what the epic's import rule
wants (siblings name the distribution root, not a private module).

The new offline example:

```python
# examples/host_report_fixture.py
# Every report rendered from RECORDED command output. No psutil, no subprocess.
inspector = HostInspector(runner=FakeRunner(results={...}))   # already exists
for report in (inspector.units(), inspector.storage(), inspector.network(), ...):
    print(render(report))
```

`FakeRunner` and `ok_result` already exist in `host/run.py:179,205`, so the
fixture example needs recorded stdout and no new seam. `metrics`/`processes` read
psutil directly and are NOT runner-injected - see open question 2.

## Sketches

Illustrative only.

```diff
# packages/host/pyproject.toml
+[project]
+name = "scufris-host"
+dependencies = ["psutil>=7.2.2", "pydantic>=2.0.0"]
+# NOT scufris-core: this package opens no database and needs no engine.
```

```diff
# scufris/api/host.py
-from ..host import HostOverview
-from ..host.overview import MIN_HOST_OVERVIEW_TTL, HostOverviewCache
-from ..metrics import Collector, HostStats
-from ..processes import ProcessCollector, ProcessList
+from scufris_host import Collector, HostOverview, HostStats, ProcessCollector, ProcessList
+from scufris_host.overview import MIN_HOST_OVERVIEW_TTL, HostOverviewCache
```

## Shape

```
        packages/host  ->  scufris_host          deps: psutil, pydantic
        +-------------------------------------+
        | run.py      Runner, run_command      |  <- the ONE subprocess seam
        | models.py   Availability, Report     |
        | units/journal/network/storage/       |
        | packages/thermal   parsers           |
        | metrics.py  Collector -> HostStats   |  <- psutil, no runner
        | processes.py ProcessCollector        |  <- psutil, no runner
        | overview.py HostOverviewCache (TTL)  |
        | render.py   report -> text           |
        | __init__.py HostInspector (facade)   |
        +-------------------------------------+
              ^                    ^                    ^
              |                    |                    |
        scufris/api/host.py   scufris/hostd/*     scufris/telegram/*
        (Stats, /api/*)       (see open q. 1)     (see open q. 3)
```

## Consequences and open questions

Cost: one distribution, and a second example file to maintain. Bought: the one
product surface that survives the v0.2.0 rewrite intact sits behind a package
with a two-line dependency list, and "this needs no privilege" becomes checkable
by reading `pyproject.toml`.

Forecloses nothing. Nothing imports host state INTO host.

**Open questions for the planner.** 1 is blocking and belongs to the epic, not
this task.

1. **`hostd` imports `host`, so this task should run BEFORE 20260803-214747.**
   Six `hostd` modules import `scufris.host.run` / `.models` / `.storage` /
   `.units` (`engine.py:33`, `preview.py:26-29`, `nixos.py:41-43`,
   `executor.py:25`, `actions/validate.py:18`, `actions/plans.py:12-13`). The
   epic's dependency table has no `hostd -> host` edge and schedules `hostd` at
   p104 ahead of `host` at p103. Recommendation: swap the priorities and amend
   the epic to `core <- host <- hostd <- hostctl`. See 20260803-214747's NOTES
   open question 1 for the rejected alternatives. Under the current order, this
   task's re-pointing work lands inside `packages/hostd` instead of `scufris/`,
   which is the same edit in a different place - so the swap costs nothing and
   the current order costs a second pass.

2. **`metrics.py` and `processes.py` take no injectable seam.** Both call
   `psutil` at module level of their collectors, and `metrics.py` also shells out
   via `subprocess` directly (it does not use `host.run.Runner`). So
   `examples/host_report_fixture.py` can render every `HostInspector` report
   offline, but CANNOT render `HostStats` without either monkeypatching psutil or
   adding a seam. Adding a seam is a behavior change and the task forbids one.
   Recommendation: scope the fixture example to the `Runner`-backed reports and
   say so in its docstring; leave the `HostStats` shape covered by the existing
   `test_metrics.py`. The DoD phrase "renders every report from fixtures" should
   be read as every `Report`, not every host-derived model.

3. **`scufris/telegram/` imports `metrics`.** `contracts.py`, `render.py` and
   `wiring.py` all use `HostStats`. Under the epic's declared graph
   `telegram <- core, chat` - no `host` edge. Telegram is still root code today
   so nothing breaks now, but the future `packages/telegram` carve will hit this.
   Not this task's problem; worth a line in the epic's Notes so it is not
   discovered twice.

4. **`test_host_digest.py`, `test_host_mcp_server.py`, `test_route_contract.py`,
   `test_app.py`, `test_domain_routers.py` import `scufris.host` but are not host
   tests.** They stay at the root and just re-point. Only the five listed under
   Surfaces move.

5. **The DoD names `test_stats_endpoint_matches_inspector_output`, which does not
   exist.** The closest existing coverage is in `tests/test_app.py` and
   `tests/test_route_contract.py`. Either write it (it is cheap - assert the
   `/api/stats` body equals `Collector().collect()` through a fake) or point the
   DoD at the existing test. Recommend writing it: it is the one assertion that
   makes "Stats still serves the same payload" falsifiable across the move.

6. **`examples/host_inspect.py` needs a marker that does not exist yet.**
   20260803-214746 builds `tests/test_examples.py` and its NOTES (open q. 7)
   recommends an explicit opt-in list rather than a marker. If the planner takes
   that, this task ADDS `host_report_fixture.py` to the list and does nothing to
   `host_inspect.py`. That is simpler than the Steps' wording and should replace it.

7. **`scripts/check_file_size.py` `COVERED_ROOTS`.** `render.py` (434) and
   `packages.py` (374) are the largest movers - under the 600-line cap, but the
   guard must already cover `packages/` from 20260803-214746 or these silently
   leave its scope.
