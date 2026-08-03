# `scufris_host` - reading the machine

The `scufris-host` distribution (`packages/host/`): read-only inspection of the
NixOS box scufris runs on. Nothing here needs privilege, nothing here changes
anything, and nothing here can - the mutating half is a separate root process
(`packages/hostd/src/scufris_hostd/README.md`).

It depends on NO sibling package, `scufris-core` included: stdlib, `psutil` and
`pydantic` is the whole list, which is what makes the epic's `host -> nothing`
edge a fact rather than an intention.

Every consumer imports the distribution ROOT and never a module inside it -
`from scufris_host import HostInspector`, never `from scufris_host.run import
Runner`. `tests/test_package_boundaries.py` enforces it. The one exception is
`render`, imported as a module (`from scufris_host import render`), which still
names the root.

`metrics.py` and `processes.py` answer "what is using the CPU right now". The
rest answers what else an operator asks: what failed, what the logs
say, what filled the disk, what is listening, whether the CPU throttled, what
provides a binary, and how the current generation differs from the last one.

## How to use it

Nothing to enable. It is always available, needs no configuration beyond
`SCUFRIS_HOST_CONFIG_REPO` (the flake it reports pin ages for), and is reachable
three ways:

```python
from scufris_host import HostInspector, Scope

inspector = HostInspector()                     # one per process is fine

report = inspector.failed_units(scope=Scope.SYSTEM)
if not report.ok:
    print("could not ask:", report.available.reason)
else:
    for unit in report.units:
        print(unit.name, unit.active, unit.sub)
```

- **The dashboard** polls `/api/host/overview`, which is `inspector.overview()`
  plus a server-side cache (`SCUFRIS_HOST_OVERVIEW_SECONDS`, default 30s). The
  overview shells out to `systemctl` and `nixos-rebuild`, so N open tabs must not
  mean N invocations.
- **Agents** reach it through the inspection MCP tools, which BOTH the
  orchestrator and the host agent hold (`scufris/mcp_host_tools/`).
- **The scheduled checks** (`scufris/checks.py`) read through it too, which is why
  `create_app(host_inspector=...)` is injectable: a real check pass walks the nix
  store, and no test should do that at import time.

`examples/host_inspect.py` prints every report against the real machine, which is
the fastest way to see what a shape actually contains.
`examples/host_report_fixture.py` prints the same tour from canned fixtures and
needs no host at all; it runs in the suite, so a renderer that rots is a red
test rather than a surprise in chat.

## Three rules that hold everywhere in here

These are why this is a package rather than a pile of shell calls.

**1. One door to the outside.** Every command goes through the `Runner` seam in
`run.py` and comes back as a classified `CommandResult`. Nothing calls
`subprocess` directly.

| `Outcome` | Means |
|---|---|
| `ok` | it ran and exited 0 |
| `missing` | the binary is not on PATH |
| `denied` | it ran and refused for lack of privilege |
| `timeout` | it exceeded its wall-clock bound |
| `failed` | any other non-zero exit |

So a missing binary, a denied permission and a timeout are three different
sentences instead of three different tracebacks. `FakeRunner` replays canned
output, which is how the whole package is tested without touching the host.

**2. Availability lives on the model.** Every report embeds an `Availability`
(`ok`, `reason`, `caveat`). A tool never raises at the MCP boundary, and an empty
result is never confusable with a broken one:

- `ok` with no rows means "asked, answered, nothing there".
- `ok is False` means "not answered, and here is what stopped it".
- `caveat` means "usable, but incomplete" - for instance socket owners that need
  privilege to resolve.

That distinction is the contract the package exists to preserve. An empty list
that reads as "nothing wrong" is the failure mode being designed against.

**3. Everything is bounded.** Journal reads, unit listings, directory walks and
closure diffs all carry caps, and a capped report says so through the `Bounded`
mixin (`truncated`, `limit`, `total_seen`, and a `truncation_marker()` renderers
must print). Defaults: 50 units per listing (max 400), 100 journal lines (max
1000, hard-capped at 40 kB), 10 second command timeout.

## What it can read

| Module | Reports | Underneath |
|---|---|---|
| `units.py` | `list_units`, `failed_units`, `unit_status` - for both the `system` and `user` scope | `systemctl list-units -o json` and `systemctl show` (stable `key=value`, never `status`) |
| `journal.py` | `read_journal` - by unit, scope, priority, time window | `journalctl` |
| `storage.py` | `filesystem_usage`, `largest_directories`, `reclaimable_space`, `list_generations`, `storage_report` | psutil (no subprocess), `du`, `nix-store --gc --print-dead`, `nixos-rebuild list-generations --json` |
| `network.py` | `list_interfaces`, `listening_sockets`, `declared_firewall`, `network_report` | `ip -j addr`, psutil for listening sockets, and the DECLARED firewall read from `/run/current-system` |
| `thermal.py` | `thermal_report` - temperatures, battery, and the CPU's cumulative throttle counters | sysfs |
| `packages.py` | `what_provides`, `profile_contents`, `closure_diff`, `flake_status` | the nix store, `nix store diff-closures`, the config repo's `flake.lock` |
| `models.py` | `Availability`, `Report`, `Bounded`, `clamp`, and the option-injection guard | - |
| `run.py` | the `Runner` seam, `CommandResult`, `Outcome`, `FakeRunner` | `subprocess` |
| `metrics.py`, `processes.py` | `HostStats` (the `/api/stats` payload) and per-application process aggregation | psutil, `nvidia-smi` |
| `inspector.py` | `HostInspector` and `HostOverview` | the modules above |
| `overview.py` | `HostOverviewCache` - one slot, a TTL floor, single-flight collection | `inspector.py` |

`HostInspector` (in `inspector.py`, so `__init__` can stay a pure re-export
door) holds the runner and the host-specific paths
(the config repo, `/run/current-system`, the CPU sysfs root) so a caller
constructs one object instead of threading four arguments through every call. It
is stateless apart from that configuration, so sharing one is safe.

`HostOverview` is the deliberately cheap subset the dashboard polls: failed units
in both scopes, storage, thermals. Every member was MEASURED cheap on this host.
`reclaimable_space` (walks the whole store) and `largest_directories` (walks a
subtree) are absent on purpose - putting either behind a poll would make the live
dashboard hostage to a store walk.

## The tools this backs

Defined in `scufris/mcp_host_tools/inspection.py` and registered by
`mcp_host_tools.register` for both the orchestrator and the host agent, since
reading needs no privilege:

`host_stats`, `disk_usage`, `list_processes`, `host_units`, `host_failed_units`,
`host_unit_status`, `host_journal`, `host_storage`, `host_largest_directories`,
`host_reclaimable_space`, `host_network`, `host_thermal`, `host_what_provides`,
`host_generation_diff`, `host_flake_status`.

## Two things it deliberately does not do

- **It never writes.** Not to the machine, and not to the config repo:
  `flake_status` reads `flake.lock` to report how old the pins are, and that is
  the whole relationship. Changing the configuration is a host action
  (`packages/hostd/src/scufris_hostd/README.md`), and editing the repo is ordinary project work.
- **It never guesses.** A value it could not read is reported as unavailable with
  the reason. `scufris/checks.py` depends on that: an UNAVAILABLE check is not a passing
  check.
