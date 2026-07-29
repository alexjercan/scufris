# Count core throttle events once per physical core

- STATUS: CLOSED
- PRIORITY: 58
- TAGS: bug, v0.2.0, host, backend

## Story

As the operator, I want the thermal card's core throttle count to be the number
of throttle events that actually happened, so that "162" is not silently double
the truth on any CPU with hyperthreading.

`read_throttling` (`scufris/host/thermal.py`) sums `core_throttle_count` across
every LOGICAL cpu under `/sys/devices/system/cpu/cpu*/thermal_throttle/`. On an
SMT machine, hyperthread siblings share a physical core and report the SAME
counter, so every core event is counted once per sibling.

Measured on this host (i9-12900F, 24 logical / SMT on, 1d12h uptime):

```
cpu8,  cpu9   core_throttle_count=16   <- one physical core (core_id 16)
cpu10, cpu11  core_throttle_count=28   <- one physical core (core_id 20)
cpu14, cpu15  core_throttle_count=37   <- one physical core (core_id 28)
the other 18 logical cpus: 0
```

Truth: 16 + 28 + 37 = **81** core throttle events across 3 physical cores.
Reported: **162**. Exactly 2x, because each core is counted on both siblings.

`core_throttle_total_time_ms` is summed the same way and is doubled identically
(reported 310ms, actual 155ms).

The same duplication one level UP was handled correctly: package counters are
identical on every cpu of a package, so `package_events` takes the MAX rather
than the sum, with a comment explaining exactly why. The reasoning was applied
to packages and not to cores.

The existing test locked the bug in. `test_throttling_sums_cores_and_takes_the_package_maximum`
builds a fixture of two cpu directories with no topology at all, so there are no
siblings to deduplicate and the sum is trivially right. A test that cannot see
the failure mode cannot pin it.

## Steps

- [x] REPRODUCE FIRST: extend the thermal fixture builder to write a
      `topology/core_id` (and `thread_siblings_list`) per cpu, and add a test
      with SMT pairs asserting the deduplicated count. Watch it fail at 2x
      before touching `thermal.py`.
- [x] Deduplicate by physical core in `read_throttling`: read
      `<cpu>/topology/core_id`, keep one counter per (package, core) rather than
      per logical cpu, and sum those.
- [x] Handle the no-topology case explicitly. A cpu directory with no
      `topology/core_id` (a container, a non-x86 host, a fixture) must still be
      counted - fall back to the cpu's own name as its identity rather than
      dropping it, so a missing topology under-counts nothing.
- [x] Report physical cores rather than logical cpus in the count that describes
      the reading: `cpus_read` currently means "logical cpus whose counter was
      readable" and is used in the rendered text ("across 24 cpus"). Decide
      whether that stays logical or becomes physical, and make the rendered
      sentence say which it is.
- [x] Re-check `core_throttle_total_time_ms`, which is summed the same way and
      doubled identically.
- [x] Verify against the live host: the example script's thermal section must
      report 81 core events, matching the raw sysfs values by hand.

## Definition of Done

- Core throttle events are counted once per PHYSICAL core, not once per logical
  cpu (test: `test_throttling_counts_each_physical_core_once`).
- The fixture used by the throttle tests has SMT sibling pairs, so the test
  would fail against the summed-per-logical-cpu implementation
  (cmd: revert the fix, watch the test go red, restore).
- A cpu directory with no topology information is still counted rather than
  dropped (test: `test_throttling_counts_a_cpu_with_no_topology`).
- The rendered thermal text says whether its cpu count is logical or physical
  (test: `test_thermal_render_names_what_it_counted`).
- cmd: `nix flake check` and `cd web && npm run ci` are green.
- manual: the thermal card's core figure matches a hand count of
  `/sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count`
  deduplicated by `topology/core_id`.

## Notes

- Epic: 20260729-124655. Fixes a defect in 20260729-125024, landed dc60a51.
- Found by the operator asking what the number meant - not by a test, not by a
  review. Two out-of-context review rounds read this code and neither caught it,
  because the arithmetic looks right and the fixture agreed with it.
- The PACKAGE half is correct and must stay: those counters are identical per
  package and `max` is the right reduction. Note that on this host they are NOT
  perfectly identical (78 on most cpus, 80 on two, 82 on two), because each cpu
  updates its own view when it handles the thermal interrupt - so `max` is also
  the right choice for freshness, not only for deduplication. Do not "fix" it to
  a sum.
- The interpretation on the dashboard card and in the tool text is sound; only
  the arithmetic is wrong. No other report is affected.

## Flow State

- FLOW STEP: DONE
- PLAN STATUS: APPROVED
