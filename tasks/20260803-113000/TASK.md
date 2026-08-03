# Prove the startup sweep clears a building row orphaned by a crash

- PRIORITY: 35
- TAGS: test,storage,nixos,v0.2.0
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -
- PARENT: 20260729-102145

## Story

As a Scufris maintainer, I want a proof that a `building` row left behind by a
process that died without running its shutdown hooks is swept at the next
startup, so that `abandon_builds` is answered by the state it actually exists
for rather than by a live process that is still building.

## Notes

- Seeded by 20260803-014401 DECISION.md 1, and required by review round 1
  (R1.1) of that task.
- `test_a_build_interrupted_by_a_restart_does_not_block_the_repo`
  (`tests/test_nixos_config_change.py:619`) is green for a reason its docstring
  does not give. Its first `TestClient` is never exited, so the hanging build
  is still hanging and the row is still `building` when the restarted app
  sweeps it. It proves the sweep clears a row some OTHER live process is
  building, not one orphaned by a dead process.
- A clean shutdown cannot produce the target state: the build generator's
  cancellation handler (`scufris/hostconfig/changes.py:329`) catches
  `CancelledError`/`GeneratorExit`, writes `state = CANCELLED` and saves BEFORE
  re-raising, so `Supervisor.aclose()` leaves `cancelled` rows, which the sweep
  neither touches nor needs to.
- Preferred shape, per that record's second alternative: let the first app's
  lifespan close normally, then re-establish a `building` row directly through
  `ConfigChangeStore` against the same state directory - that is what a SIGKILL
  leaves - and build the restarted app. Comment WHY the clean path cannot
  produce the row, so the next reader does not "simplify" it back to an HTTP
  build.
- Keep `test_a_build_interrupted_by_a_restart_does_not_block_the_repo` as is;
  it covers the live-process case and the repo-unblocked follow-through. This
  is an addition, not a replacement.
- Scope is tests only. `abandon_builds` is already correct; no production
  change is expected.
