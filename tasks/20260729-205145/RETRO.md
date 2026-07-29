# Retro: count core throttle events once per physical core

- TASK: 20260729-205145
- EPIC: 20260729-124655
- REVIEW ROUNDS: 2 (REQUEST_CHANGES, APPROVE)
- FIXES: a defect in 20260729-125024, landed dc60a51

## What went well

**The reproduction was written first and reproduced the real number exactly.**
The fixture replays this host's actual sysfs shape - cores 16/20/28 at 16/28/37
events, each on both siblings - so the failure was `assert 162 == 81`, the same
162 the dashboard had shown. Not an approximation of the bug; the bug.

**Reverting the fix proved the pin, and proved it for the right reason.** The
reviewer independently re-ran the revert and confirmed the test trips on the
doubled count before it reaches any newly added field - so it is a real
reproducer, not a test that happens to fail because the model changed.

**The fix was verified against a hand count, not against itself.** The DoD asked
for the tool's figure to match a manual dedup of the raw sysfs; both say 81
across 16 cores. A test using the same helper as the implementation would have
proved much less.

## What went wrong

**The bug was found by the operator asking what a number meant.** Not by a test,
not by two out-of-context review rounds that read this exact function. It
survived because the arithmetic *looked* right and the fixture agreed with it -
`sum(core_throttle_count)` over cpu directories reads as obviously correct until
you know that siblings share the counter.

**I had already reasoned about this duplication one level up.** The package
counters take `max` precisely because they are duplicated per cpu, with a comment
explaining why. I applied the insight to packages and not to cores, in the same
function, in the same loop. Having the right idea in scope is not the same as
applying it everywhere it holds.

**The original test locked the bug in.** `test_throttling_sums_cores_and_takes_the_package_maximum`
built two cpu directories with no topology at all - so there were no siblings to
deduplicate, the sum was trivially right, and the test blessed the wrong
behaviour. A fixture that cannot express the failure mode cannot pin against it,
and its passing is evidence of nothing.

**And I repeated the same class in the fix.** Review round 1 caught two more:
siblings reduced by last-write-wins rather than `max` (the identical reasoning
gap, one more time), and "81 across 16 cores" quoting a per-core figure against
the count of all cores - the very misreading this task exists to prevent, one
step removed.

**A DoD I wrote named a test I then did not write.** `test_thermal_render_names_what_it_counted`
was in the Definition of Done and simply absent; the frontend wording got a test
and the Python renderer did not. The DoD is not a checklist that verifies itself.

## What to do differently

1. **When a value is read per-cpu, ask what hardware it actually belongs to**
   before aggregating: per-thread, per-core, per-package and per-socket all look
   identical in `/sys/devices/system/cpu/cpu*/`. The directory layout is a lie
   about the topology.
2. **When you apply a de-duplication in one place, sweep for the same shape in
   the same function.** The package `max` and the core sum were four lines apart.
3. **Build the fixture from the real thing's SHAPE, not just its values.** The
   original fixture had the right numbers and the wrong structure - no siblings,
   no topology - which is why it agreed with a doubled total.
4. **Re-read the DoD before opening a review**, and check each named test exists.
   Both misses this round were things I had already written down.

## Lessons for the ledger

- `sysfs-per-cpu-counters-are-not-per-cpu-quantities` (Monitoring/collector)
- `a-fixture-that-cannot-express-the-bug-blesses-it` (Testing)
