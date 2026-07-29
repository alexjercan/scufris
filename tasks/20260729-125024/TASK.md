# Expand read-only host inspection beyond stats

- STATUS: CLOSED
- PRIORITY: 60
- TAGS: feature,v0.2.0,host,mcp,backend

## Story

As the operator, I want to ask the assistant real questions about this machine
and get answers from typed tools, so that "why is this box hot", "what filled
the disk", and "did anything fail overnight" are answerable in chat.

Today the host toolset is `host_stats`, `disk_usage`, and `list_processes`. That
covers "what is using memory" and nothing else. This task is pure read-only
expansion: it is useful on its own, it is the input every later mutating action
reasons from, and it needs no privilege decision to ship.

## Steps

- [x] Lay the `scufris/host/` package foundation: a `Runner` seam (injectable,
      like `metrics.GpuRunner`) returning a typed `CommandResult` whose outcome
      is one of ok / missing / denied / timeout / failed, plus the shared
      `Availability` model every report embeds. Nothing above this layer calls
      `subprocess` directly, so "degrades explicitly" is a property of the layer
      rather than a habit each tool has to remember.
- [x] Add systemd inspection (`host/units.py`): unit list with state filtering,
      one unit's status and last invocation result, failed units, and user vs
      system scope. Parse `systemctl -o json` / `systemctl show` key=value, not
      the human table.
- [x] Add journal query (`host/journal.py`): bounded reads by unit, priority and
      time window over `journalctl -o json`, with a line cap AND a byte cap, an
      explicit truncation marker, and both `--user` and system scope.
- [x] Add storage inspection (`host/storage.py`): filesystem usage by mount
      (psutil), Nix store size, system generations with dates and versions from
      `nixos-rebuild list-generations --json`, largest directories under a
      bounded root, and garbage-collectable space from
      `nix-collect-garbage --dry-run`.
- [x] Add network inspection (`host/network.py`): interfaces and addresses from
      `ip -j addr`, listening sockets from psutil, and the DECLARED firewall
      parsed from `/run/current-system`'s `firewall-start` script.
- [x] Add thermal and power (`host/thermal.py`): per-zone and per-core
      temperatures, GPU temperature, and the `thermal_throttle` sysfs counters
      that actually answer "is it throttling". Battery and fans are implemented
      as explicit not-present results on this host, not omitted.
- [x] Add package and generation queries (`host/packages.py`): what provides a
      binary (PATH -> realpath -> store path -> derivation name), what is in the
      profiles, `nix store diff-closures` between two generations, and whether
      the config repo's flake inputs are behind.
- [x] Wire the MCP tools in `mcp_server.py` with docstrings that say when to
      PREFER each over shell, the way `list_processes` already does, and render
      each report as compact text with an explicit `unavailable: <reason>` line
      rather than a blank section.
- [x] Add `GET /api/host/overview` returning the cheap, glanceable subset behind
      a short TTL cache, and extend the stats page with FAILED UNITS,
      GENERATIONS, NIX STORE and THERMAL cards polled on their own slower
      interval (never on the 2s stats poll).
- [x] Add `examples/host_inspect.py` that prints every report against the real
      host, as the end-to-end proof and a re-runnable probe rig.

## Definition of Done

- Units, logs, storage, network, sensors and generations are readable through
  typed tools with bounded output
  (test: `test_host_inspection_covers_units_logs_and_storage`).
- Every tool degrades to an explicit unavailable/unsupported result instead of
  raising or returning a misleading empty value
  (test: `test_host_inspection_tools_degrade_explicitly`).
- Journal and directory reads cannot exceed their configured caps
  (test: `test_host_inspection_output_is_bounded`).
- A successful-but-empty result is never rendered as a blank: an empty closure
  diff says "no closure change", no failed units says "none failed", and a
  socket whose owner is invisible without privilege says so
  (test: `test_host_inspection_distinguishes_empty_from_broken`).
- The dashboard shows the overview cards and renders an unavailable report as a
  visible reason (test: `renders host cards` / `renders an unavailable host
  report with its reason` in `web/src/stats-view.test.ts`).
- The overview endpoint serves the cached snapshot rather than re-shelling per
  request (test: `test_host_overview_is_cached`).
- cmd: `nix flake check` and `cd web && npm run ci` are green.
- manual: asking the orchestrator "why is this box hot" and "what filled the
  disk" produces a specific, correct answer without a terminal.

## Notes

- Epic: 20260729-124655.
- Read-only: nothing in this task changes system state, so it can land before
  the privilege decision from the host spike.
- SPIKE OUTCOME (`tasks/20260729-125020/DECISION.md`): confirmed and stronger
  than assumed. Every inspection this task needs was MEASURED working as `alex`
  with no privilege at all - `journalctl -u <system unit>` (wheel gets the
  journal ACL), `nixos-rebuild list-generations`, `nix store diff-closures`,
  `systemctl show`. So this task needs no helper, no sudo, and no waiting: it
  can run in parallel with the privileged machinery in 20260729-125029.
- Use `nix store diff-closures` (builtin to nix 2.34.8 here), not `nvd` - `nvd`
  is not installed and the builtin means no new dependency in `nix.dotfiles`.
- Build on `scufris/metrics.py` and `scufris/processes.py` conventions
  (psutil-backed, structured records, tolerant of missing data).
- The MCP tool docstrings are prompt surface: say when to PREFER a tool over
  shell, the way `list_processes` already does.

### Measured on this host at plan time (2026-07-29, as `alex`, no privilege)

Re-probed rather than inherited, per `probe-runtime-on-target-host-early`.

- Machine-readable output exists for everything that matters and is what the
  parsers target: `systemctl list-units -o json`, `journalctl -o json` (system
  AND `--user`), `nixos-rebuild list-generations --json`, `ip -j addr`. No
  human-table scraping.
- `nix store diff-closures` between generations 190 and 191 printed NOTHING and
  exited 0 - the spike's trap, reproduced live. The renderer must branch on exit
  status and say "no closure change" explicitly.
- **`iptables -L` is root-only here** ("Could not fetch rule set generation id:
  Permission denied"), so the LIVE rule set is not readable as `alex` and the
  step's "current firewall rule state" cannot be delivered as written. The
  DECLARED set is readable: `/run/current-system`'s `firewall-start` script
  lists every `ip46tables -A nixos-fw ... --dport N -j nixos-fw-accept`. Report
  that, labelled "declared", and say the live set needs privilege - the same
  honesty rule the spike applied to previews.
- **This host is a DESKTOP, not the laptop the step assumed**: `chassis_type=3`,
  `/sys/class/power_supply/` is empty, `psutil.sensors_battery()` is `None`,
  `psutil.sensors_fans()` is `{}`. Battery and fans therefore land as explicit
  not-present results (operator-confirmed at the plan gate). The signal that
  actually answers "why is it hot" is `coretemp` plus
  `/sys/devices/system/cpu/cpu*/thermal_throttle/*` - this box has recorded 78
  package throttle events, which is a real finding a temperature gauge misses.
- Listening-socket ownership is PARTIAL as `alex`: psutil sees 18 listeners and
  resolves a pid for 7. Owner-unknown rows must say "owner not visible without
  privilege", never render blank.
- `nix profile list` shows only `home-manager-path`; the 1155 binaries in
  `/run/current-system/sw/bin` are the system profile. Both are worth reporting.
- Cost measurements for the poll decision: failed units 0.00s, generations
  0.11s, all units 0.00s - cheap enough for the overview. But
  `nix-collect-garbage --dry-run` walks the whole store (7974 dead paths here)
  and is NOT poll-safe, so reclaimable space stays an on-demand agent tool and
  is deliberately absent from `/api/host/overview`.

### Design decisions taken at the plan gate

- **`scufris/host/` package, not one module.** Six domains with distinct parsers
  and distinct failure modes; a single file would be a grab bag.
- **One `Runner` seam for the whole package.** Tests inject a fake runner
  replaying REAL captured output (`capture-real-cli-output-for-parser-tests`),
  so no test patches `subprocess` internals and every degrade case (missing
  binary, denied, timeout, non-zero exit) is exercised through the same door.
- **Availability is on the model, not in a raised exception.** Every report
  embeds `Availability(ok, reason)`; a tool never raises at the MCP boundary and
  never returns a bare empty list that reads as "nothing wrong".
- **The stats page gets cards, not a new page** (operator's call at the gate).
  They come from a separate `/api/host/overview` on a slower interval with a
  server-side TTL cache, because folding subprocess calls into the 2s
  `/api/stats` poll would make the live dashboard hostage to `nixos-rebuild`.
- **Journal and largest-dirs stay agent-only.** They are query-shaped (which
  unit, which window, which root) and have no glanceable form.

## Flow State

- FLOW STEP: DONE
- PLAN STATUS: APPROVED
