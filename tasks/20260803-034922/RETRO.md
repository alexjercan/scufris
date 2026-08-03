# Retro: pin the two legacy-diagnostics tests that cannot go red

- TASK: 20260803-034922
- BRANCH: fix/pin-legacy-diagnostics-tests
- REVIEW ROUNDS: 1 (APPROVE)

## Falsification transcript

`bash tasks/20260803-034922/falsify.sh` -> exit 0, "both sabotages falsify
their test". Per sabotage:

### R2.1 - `sabotage-r21.patch` (restores the `agent_enabled` short-circuit)

RED under sabotage:

```text
FAILED tests/test_app.py::test_disabled_agent_is_supported_not_unsupported
>       assert usage["value"]["primary"]["used_percent"] == 42.0
E       TypeError: 'NoneType' object is not subscriptable
```

GREEN once reverted: `1 passed`.

The short-circuit returns `Capability.read(None)` for usage, an all-zero
`MemoryFootprint` for memory, and an empty quota on the account. The rewritten
test asserts the DELEGATED reading (42.0% primary usage, one session, populated
account quota), which only the real readers can produce - so the short-circuit
can no longer hide behind an empty home.

### R2.2 - `sabotage-r22.patch` (drops the `quota.value` unwrap)

RED under sabotage:

```text
FAIL  src/agent-view.test.ts > legacy /api/agent/usage capability envelope
      (startAgent) > renders the meter from a supported envelope's value
AssertionError: expected true to be false // Object.is equality
 Test Files  1 failed (1)
      Tests  1 failed | 5 skipped (6)
```

GREEN once reverted: `Tests  1 passed | 5 skipped (6)`.

The harness targets the SURVIVING positive case, not the deleted one: that case
was already falsifying before this task. What was removed is the vacuous
negative twin (DECISION.md); what was added is the harness that keeps the pin
honest.

## Suites

- `python -m pytest` -> exit 0.
- `cd web && npm run ci` -> exit 0 (format:check, lint, test, build).

## What went well

- Building the harness before touching either test paid immediately: it came
  back red on R2.1 and already green on R2.2, which split a "two broken tests"
  brief into one rewrite and one deletion instead of two speculative rewrites.
- TASK.md's Notes had already caught that `5444fa1` split `scufris/app.py` into
  routers, so the r21 sabotage was authored against
  `scufris/api/legacy_agent.py` without a detour.

## What went wrong

- The first harness draft drove vitest through `npm run test --`. npm re-split
  the `-t` pattern into separate words, and the run failed with
  `vitest: command not found` - which the harness scored as "red under
  sabotage". A false green was one step away; only the paired
  restore-and-expect-green check exposed it.
- The sprout worktree had no `web/node_modules`, so the frontend half of the
  harness could not run until `npm ci`. Nothing in the harness said so.
- `get_account` has no `settings`-shaped seam left after the router split, so
  the sabotage rewrites the quota on the returned `AccountInfo` rather than
  restoring the original branch verbatim. Same observable envelope, but the
  patch is a behavioural stand-in, not a literal revert.

## What to improve next time

- A harness that asserts "this command fails" must distinguish a failing TEST
  from a failing HARNESS. The revert-and-expect-green half is what caught it
  here; keep both halves rather than trusting a single red.
- Invoke test binaries directly (`node_modules/.bin/<tool>`) when an argument
  contains spaces. `npm run <script> --` is not argument-preserving.
- Preflight the environment a proof script needs, and fail with the fix
  (`cd web && npm ci`) rather than with a tool's own error.

## Diagnosis

- **Breadth.** The diff is small (two test files, one script, two patches) and
  matches the plan's four Steps one for one. No split was missed: the two
  findings share a single artifact, the harness, so landing them separately
  would have built it twice.
- **Churn.** Zero review rework - Round 1 approved. The plan-time question that
  earned that was building the harness FIRST, which is what turned R2.2 from an
  assumed rewrite into an evidenced deletion before any test was touched. The
  one plan defect a from-scratch challenge would have caught earlier is the
  stale `git show master:scufris/app.py` anchor; TASK.md's Notes caught it at
  work time instead, at the cost of re-locating the sabotage target.
- **Context.** No threshold crossing, compaction warning or checkpoint was
  recorded. Round 1 was delegated to an out-of-context reviewer as the flow
  requires; nothing else needed splitting or deferring.

## Review outcome

Round 1, out-of-context reviewer, APPROVE with three open non-blocking
findings (R1.1 MINOR, R1.2 MINOR, R1.3 NIT), all against the harness itself
rather than the pins it proves. Disposition: accepted as-is, no follow-up task.

- R1.1 (no cleanup trap) and R1.3 (RED phase accepts any non-zero exit) harden
  a script that already failed loudly on every path exercised here, including
  the false red that R1.2's sibling lesson records. The remaining risk is an
  interrupted run leaving a sabotage applied, which `git status` surfaces
  immediately.
- R1.2 asks for the r21 sabotage to be split per reader, since pytest aborts at
  the first failing assertion and only the usage one is mechanically proven.
  The discrimination for the other two is already evident in the diff: the old
  test asserted exactly the readings the short-circuit produces
  (`session_count == 0`, `quota value None`), and each assertion still fires
  independently against a single-reader regression.

## Action items

- None. No product code changed; R2.1 and R2.2 from 20260801-100415's Round 2
  are both closed by this task, and Round 1's findings are accepted above.

## Landing message

```text
test(agents): make the two legacy-diagnostics pins falsifiable

Round 2 of 20260801-100415 flagged two tests that passed with or without the
behaviour they name. Add a falsification harness under tasks/ - two sabotage
patches and falsify.sh, which applies each, requires the named test red,
reverts, and requires it green - then fix what it exposed.

test_disabled_agent_is_supported_not_unsupported now seeds a populated codex
home and asserts the delegated reading, which the agent_enabled short-circuit
cannot produce. The frontend's "hides the meter when the backend cannot report
usage" case is deleted per DECISION.md: renderUsage empties and hides the meter
for every primary-less shape, so no assertion over it can discriminate; the
surviving positive case is the quota.value unwrap's pin.

No product code changed.
```
