# Decision: A graceful shutdown is not a crash, so the interrupted-build proof is left alone

- DATE: 20260803-014401
- STATUS: ACCEPTED
- TASK: 20260803-014401
- TAGS: test,storage,nixos

## Context

Step 2 assumed the first app's lifespan shutdown "cancels the in-flight run
through `runs.aclose()` and leaves the row `building`", which is the state
`abandon_builds` exists to sweep. It does not.

The build generator's cancellation handler
(`scufris/hostconfig/changes.py:329`) catches `CancelledError`/`GeneratorExit`,
sets `state = CANCELLED` and `await save(...)` BEFORE re-raising. So a run
cancelled by `Supervisor.aclose()` (`scufris/supervisor.py:300`, called from
`scufris/app.py:236`) is written back as `cancelled`, not left `building`.

Restructuring `test_a_build_interrupted_by_a_restart_does_not_block_the_repo`
the way Step 2 describes makes it red for that reason rather than for the
identity assertion the task is about:

    assert swept["state"] == "failed"
    AssertionError: assert 'cancelled' == 'failed'

Step 3 named this outcome in advance and said to stop and record it rather than
patch around it.

## Decision

**Leave that test byte-identical to the base.** Only
`test_a_configuration_change_survives_a_restart` gets the reopen and the
`state.db is not` assertion. The DoD clause covering both restart proofs is met
for one of them.

What the finding means beyond the test:

- The existing test passes for a reason its docstring does not give. The row is
  `building` at the restarted app's startup only because the first app was never
  shut down, so its hanging build was still hanging. It proves the sweep clears
  a row some OTHER live process is still building, not one left behind by a
  process that died.
- `abandon_builds` is still correct and still needed. The state it exists for is
  reachable only from a process that dies WITHOUT running its shutdown hooks -
  SIGKILL, OOM, power loss. A clean `systemctl restart` leaves `cancelled` rows,
  which the sweep does not touch and does not need to.

## Alternatives considered

- **Weaken the assertion to accept `cancelled`.** Rejected: it destroys the
  property. `cancelled` is the operator-stopped state; a sweep that answers it
  proves nothing about a crash.
- **Re-establish a `building` row through `ConfigChangeStore` after the clean
  shutdown, commented with why the clean path cannot produce one.** Plausible
  and probably right, but it is a different test from the one planned and a
  decision about what the proof should MEAN, not an implementation detail. It
  belongs to whoever decides that, not to this task's Step 2.
- **A first app whose lifespan never runs.** Would model the crash faithfully,
  but the build only runs inside the lifespan, so there is no in-flight build to
  interrupt.

## Consequences

- The gap is now written down instead of hiding behind a green test whose
  docstring overclaims. Modelling a real crash is task 20260803-113000, seeded
  from this record and filed under the same epic: re-establish a `building` row
  through `ConfigChangeStore` after a clean shutdown, per the second
  alternative above.
- R1.3 is taken as planned: `abandon_builds` returns `None`, its one caller
  (`scufris/app.py:423`) already discarded the value. Startup observability, if
  wanted, is its own task.
