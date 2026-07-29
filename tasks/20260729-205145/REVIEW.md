# Review: count core throttle events once per physical core

- TASK: 20260729-205145
- BRANCH: fix/throttle-per-core

## Round 1 - REQUEST_CHANGES

- VERDICT: REQUEST_CHANGES

Out-of-context reviewer against the working tree on top of 11bd70c. The central
fix was confirmed correct against the live host (81 events / 16 cores, matching a
hand count of the raw sysfs), and the reproduction was confirmed to fail against
the old implementation for the RIGHT reason - `assert 162 == 81`, tripping on the
doubled count before it reaches any new field. Five findings, two MEDIUM.

### MEDIUM 1 - siblings reduced by last-write-wins, not max

The dict overwrite kept whichever sibling was globbed last, and `sorted()` is
lexicographic, so "cpu10" precedes "cpu2" and the winner is arbitrary. The
reviewer's argument is the one already written into this module for the package
counters: these are written by each cpu's OWN thermal-interrupt handler, so two
siblings can be momentarily out of step - and the package counters prove that
skew is real here (78 on most cpus, 80 on two, 82 on two, exactly the cpus that
also carry non-zero core counts). The same reduction belongs one level down.

**Fixed**: `max` per field. Pinned by a test with a disagreeing pair (37 vs 36)
that fails under last-write-wins.

### MEDIUM 2 - a Definition-of-Done test did not exist

The DoD names `test_thermal_render_names_what_it_counted`; it was never written,
so both new sentences were entirely unpinned and only the FRONTEND wording had a
test. The Python renderer's phrasing is half of what this task is about - "162
core events across 24 cpus" was the defect's second half, and fixing only the
arithmetic would still leave a figure quoted against the wrong denominator.

**Fixed**: the named test now asserts both branches - the throttled sentence says
"per-core events", "whole-package events" and "3 of 16 physical cores", and the
quiet sentence distinguishes physical cores from logical cpus. It also asserts
the ambiguous old phrasings are ABSENT.

### LOW 3 - an unreadable socket id merged cores across sockets

`_core_identity` was careful never to drop a cpu when `core_id` was missing, but
`physical_package_id` fell back to `0`, so a readable core_id with an unreadable
package id collapsed every socket into one - merging core 0 of socket 0 with core
0 of socket 1. An UNDERCOUNT, the exact direction the fallback exists to prevent.

**Fixed**: either id missing falls back to the cpu's own name, which over-counts
at worst. Two tests: a real 2-socket layout stays two cores, and a missing
package id does not merge.

### LOW 4 - "across 16 cores" invited the wrong denominator again

Only 3 of 16 cores ever throttled, and "81 across 16 cores" reads as a
distribution over 16 - the same class of misreading this task exists to fix, one
step removed. **Fixed**: `cores_throttled` is now tracked and both surfaces say
"on 3 of 16 cores", which is also the more interesting figure (the concentration
is the finding).

### NIT 5 - a vacuous assertion

`expect(text).not.toContain("81 core / 82 package")` cannot fail once the four
positive assertions above it hold. Removed.

### Checked and clean

`core_time_ms` reduced identically to `core_events`; the package `max` untouched
and now tested with genuinely divergent values; no remaining place says "cpus"
about a per-core number; `HostOverview` serialises the new fields straight from
pydantic and `common.ts` matches.

## Round 2 - APPROVE

- VERDICT: APPROVE

All five addressed. Gates green by EXIT CODE (not a piped tail):
`nix flake check` 0, `nix build .#scufris .#web` 0, `npm run ci` 0.

Verified live after the fixes: "81 per-core events (155ms total) on 3 of 16
physical cores, and 82 whole-package events (153ms)", matching a hand count of
`/sys/devices/system/cpu/cpu*/thermal_throttle/` deduplicated by `core_id`.
