# Implementation notes: read-only host inspection

- TASK: 20260729-125024
- EPIC: 20260729-124655
- BRANCH: feat/host-inspection

## What shipped

A `scufris/host/` package (9 modules, ~1900 lines with docs), 12 MCP tools, one
cached HTTP endpoint, four dashboard cards, a runnable example and 26 tests over
fixtures captured from the real machine.

| Module | Answers |
|---|---|
| `run.py` | The single door to every CLI: a `Runner` seam returning a `CommandResult` classified ok / missing / denied / timeout / failed. |
| `models.py` | `Availability` (ok + reason + caveat), `Bounded` (truncation), `clamp`. |
| `units.py` | systemd units: listing, one unit's status, failed units, system vs user scope. |
| `journal.py` | Bounded journal reads by unit, priority, time window. |
| `storage.py` | Filesystems, the Nix store, generations, largest directories, reclaimable paths. |
| `network.py` | Interfaces, listening sockets, the DECLARED firewall. |
| `thermal.py` | Temperatures, throttle counters, battery, fans. |
| `packages.py` | What provides a binary, profiles, closure diffs, flake pin ages. |
| `render.py` | Compact text for the agent, with the empty-vs-broken distinction enforced. |

## Why it is shaped this way

**One runner seam, not per-module subprocess calls.** The task's third DoD item
("degrades to an explicit unavailable result instead of raising or returning a
misleading empty value") is a property that decays if every module has to
remember it. Routing everything through `run.py` makes classification structural:
a module physically cannot get output without also getting the outcome that
explains it. It also makes every failure mode testable through the same door as
the success path - `test_host_inspection_tools_degrade_explicitly` drives four
failure modes across eight reports through one `FakeRunner`.

**Availability on the model, not exceptions.** An MCP tool that raises hands the
model a traceback instead of an answer. Every report embeds
`Availability(ok, reason, caveat)`; `test_unavailable_inspections_never_raise`
drives all 16 entry points against a host with nothing installed.

The third state, `caveat`, earned its place during implementation: several
reports are genuinely PARTIAL rather than either fine or broken (46 of 54 sockets
have an invisible owner; `du` exits non-zero for one unreadable subdirectory
while printing everything else; the firewall is declared rather than live).
Collapsing those into ok/not-ok would have meant either discarding real data or
presenting it without its qualification.

**Machine-readable output only.** `systemctl -o json`, `journalctl -o json`,
`nixos-rebuild list-generations --json`, `ip -j addr`. No human-table scraping:
column widths are not a contract.

**Two clocks on the dashboard.** `/api/host/overview` runs subprocesses, so it
has its own interval (30s) and a server-side TTL cache. The 2s `/api/stats` poll
is untouched. Measured before deciding: failed units 0.00s, generations 0.11s -
cheap enough to poll; `nix-collect-garbage --dry-run` walks the whole store
(7974 dead paths here) - excluded from the overview entirely and left as an
on-demand agent tool.

## Ground truth measured on the host (2026-07-29)

Re-probed rather than inherited from the spike. Three findings contradicted the
task text as written and changed what shipped:

1. **`iptables -L` is root-only here.** The live rule set is not readable as
   `alex`, so "the current firewall rule state" could not ship as specified.
   What IS readable is the declared set, from the `firewall-start` script the
   activated generation references. It ships labelled DECLARED with an explicit
   note that the live table needs root - the same honesty rule the spike applied
   to previews. `FirewallReport.declared_only` is a field, not just prose, and a
   test asserts the label reaches the rendered text.
2. **This host is a desktop, not the laptop the task assumed.**
   `chassis_type=3`, `/sys/class/power_supply/` empty, `sensors_battery()` None,
   `sensors_fans()` `{}`. Battery and fans ship as explicit not-present results
   (operator's call at the plan gate). The signal that actually answers "why is
   it hot" turned out to be the throttle counters: this box has recorded 162 core
   and 82 package throttle events while reading 71C - a real finding no
   temperature gauge surfaces.
3. **The closure-diff trap reproduced exactly.** `nix store diff-closures`
   between generations 190 and 191 printed nothing and exited 0. Handled by
   branching on exit status into `identical=True`, with a test that the failing
   case and the identical case render differently.

Also measured: socket ownership is partial (18 listeners, 7 with a resolvable
pid); the system profile is 1155 binaries while `nix profile list` shows only
`home-manager-path`.

## Bugs found, and how

All three of the interesting ones were found by **running
`examples/host_inspect.py` against the real host**, not by tests. The example
paid for itself before it was committed.

1. **`psutil.SOCK_STREAM` does not exist** (it is `socket.SOCK_STREAM`). An
   AttributeError on the first live run. Invisible to any test that faked the
   socket table.
2. **The store-path regex swallowed the binary path.** `what_provides("systemctl")`
   reported the package as `systemd 261/bin/systemctl`. Cause: the pattern is
   matched against every ancestor of the resolved binary *starting with the
   binary itself*, and `(?P<name>.+?)$` matches a path with slashes in it, so the
   first candidate won. Fixed to `[^/]+`. The regression pin asserts the NEGATIVE
   case (the pattern must REFUSE `.../bin/systemctl`); the positive assertion
   alone passes with the buggy pattern, which is the trap
   `dod-named-tests-deserve-the-most-scrutiny` warns about.
3. **Duplicate firewall ports.** 11433 is opened both globally and
   per-interface, so the summary line listed it twice, reading as two openings.
   Deduplicated in declaration order.

Two more came out of the test suite itself:

4. **`isinstance(parsed, str)` classified every valid scope as an error.**
   `Scope` is a `StrEnum`, so a returned scope IS a `str` - the error-sentinel
   pattern was broken for the success path, and `host_units()` returned the enum
   instead of a report. Caught by a parametrized tool test asserting the shape of
   the return value; a test that only checked the error path would have passed.
   Fixed by using `None` as the failure signal.
5. **The failed-units card showed "0 failed" over an unread scope.** Rendering
   the readable scope's total when the other scope failed is precisely the false
   reassurance these cards exist to prevent. Now the count is only shown when
   EVERY scope was read; otherwise "?" plus per-scope rows. Caught by the
   frontend test asserting "?" - it failed on the first run and was right to.

Both fixes 2 and 3 were verified by reverting them and watching the tests go red
(`revert-the-fix-to-prove-the-test`), then restoring.

One pre-existing guard also fired usefully: `test_openapi_docs_are_organized`
refused the new route for having no OpenAPI tag, which is exactly the job a
route-enumerating test exists to do.

## What I would do differently

- **Run the example earlier.** It was written after the six domain modules, and
  it found three bugs in its first run. Writing the end-to-end script alongside
  the FIRST module - even printing one report - would have caught the socket bug
  before five more modules were built on the same assumption.
- **The `Scope` StrEnum sentinel bug was avoidable by reading the codebase's own
  convention.** `enums.py` documents plainly that every member IS its string
  value; the `isinstance(x, str)` sentinel is incompatible with that by
  construction, and the module docstring said so before the bug was written.
- **The `caveat` third state should have been in the plan.** It was added
  mid-implementation once partial reports appeared in three separate modules. The
  plan's binary ok/unavailable model was too coarse for the data, and that was
  discoverable at plan time by looking at the socket-ownership measurement I had
  already taken.
