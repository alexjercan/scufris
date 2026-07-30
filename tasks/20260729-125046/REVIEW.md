# Review: Add scheduled host checks and a proactive digest

- TASK: 20260729-125046
- BRANCH: feat/host-digest

## Round 1

- VERDICT: REQUEST_CHANGES
- REVIEWER: in-session (no out-of-context mechanism available - subagents are
  disabled in this session, so the round-1 default could not be used. Recorded as
  the exception the review skill allows. The MAJOR below came from PROBING the
  renderer over repeated ticks, and the probe output is quoted.)

What was verified rather than taken on trust:

- `python -m pytest` 879 passed; `cd web && npm run ci` 258 vitest tests plus the
  webpack build; `nix flake check` all four checks.
- The daily cadence was probed over three simulated days of ten-minute ticks: it
  fired exactly once each day at 08:00 (`Mon 08:00', 'Tue 08:00', 'Wed 08:00`), so
  the time-of-day arithmetic and the re-arming are right.
- `examples/host_digest.py` was run and READ in all five states, which is what
  caught the detail-placement bug fixed before this review (the disk's filesystem
  lines were printing under the store's headline).
- The build-time correction is pinned: nothing fires on a fresh schedule
  (`test_the_scheduler_is_started_by_the_app`), which is what stopped every app boot
  in the suite from reading the real host.

- [x] R1.1 (MAJOR) scufris/digest.py `render_digest` - a condition that PERSISTS is
  re-sent every interval, which is precisely how this feature gets muted. `watch`
  renders whenever `attention` is non-empty, and "the disk is 96% full" is non-empty
  every fifteen minutes until someone fixes it. Probed over four ticks of one
  unchanged crit: `PROBE messages for ONE unchanged condition over 4 ticks: 4` -
  "that is 96 a day for a disk that has not moved". The task's own note says a noisy
  digest gets muted and a muted digest is a deleted feature, and DECISION.md section
  2 says `watch` sends when there is NEWS - a condition the operator was told about
  an hour ago is not news. Suggested change: `watch` renders only when something
  CHANGED (a check that entered or worsened an attention state, or recovered);
  persistent state is what the daily line is for. Pin it with the probe as a test:
  four ticks of an unchanged crit produce exactly one message, and a worsening
  (warn -> crit) produces another.
  - Response: fixed. `render_digest` computes what CHANGED against the last digest's
    states and returns None for `watch` when nothing did; `daily` still renders
    unconditionally, which is where a standing condition gets repeated. Pinned by
    `test_an_unchanged_condition_is_not_re_sent` - the probe as an assertion: four
    ticks of one unchanged crit produce exactly one message, a worsening produces
    another, a recovery produces another, and the daily line still repeats the
    standing condition.
- [x] R1.2 (MAJOR) scufris/app.py `_escalate_breaches` - the same repetition, with
  worse consequences: a breached check proposes a store collection on EVERY run while
  the breach lasts. With `check_escalate_gc` on and a full store that is a new pending
  root-action proposal every fifteen minutes, each one announcing itself to Telegram
  and to the queue, until the helper's per-requester cap starts refusing. Suggested
  change: escalate only when the check's state CHANGED into the breach, and skip
  entirely while an equivalent proposal from `scheduled-check` is still decidable
  (`approvals.decidable()`) - one pending collection is the ask; a second is noise.
  Pin both.
  - Response: fixed, both guards. A check escalates only when its state changed into
    the breach, and never while an equivalent proposal from `SCHEDULED_CHECK_ACTOR` is
    still decidable. Pinned by `test_a_standing_breach_is_not_re_escalated`: four
    passes over the same breach leave exactly one pending proposal, and nothing ran.
- [x] R1.3 (MINOR) scufris/digest.py:155 - `DigestStore.__init__` takes a `clock` it
  never reads (probed: `DigestStore accepts a clock it never reads: True`). A
  parameter that suggests injectable time where there is none is a small lie in the
  API; drop it.
  - Response: dropped, along with the now-unused `time` and `Callable` imports.
- [x] R1.4 (MINOR) scufris/checks.py `run_checks` - the `only` parameter is not used
  by anything (probed: it is in the signature, no caller passes it). Speculative
  generality on the function that every check goes through; drop it until something
  needs a partial run.
  - Response: dropped, with its `Iterable` import and the filter branches that only
    existed to serve it.
- [x] R1.5 (NIT) tests/test_host_digest.py `test_digest_survives_a_failing_check` -
  the local `hang` helper is defined and never used (the timeout is exercised by
  `sleeper`). Dead scaffolding inside the test that pins the failure path.
  - Response: removed.

Pending user checks (not resolved by this review):

- manual: after a week of daily digests, the operator still reads them. Nothing in
  this session can answer that - it needs a week and the deployment. What is
  available now is `examples/host_digest.py`, which prints all five states and is the
  fastest way to judge the wording before living with it.

## Round 2

- VERDICT: APPROVE
- REVIEWER: in-session (same exception as round 1. Each fix re-verified by running its
  pin; both MAJORs re-derived by replaying their probes as assertions rather than by
  reading the patch.)

All five findings are resolved and ticked.

- R1.1 re-verified as an assertion: the four-tick probe now expects exactly one
  message, and reverting the change fails it - the state-keyed version sent four.
- R1.2 re-verified the same way: four passes over one standing breach leave one
  pending proposal, and the executor is untouched throughout.
- R1.3, R1.4 and R1.5 are removals; ruff and mypy over 115 files confirm nothing
  referenced them.

Gate after the fixes: `python -m pytest` 881 passed; `nix flake check` all four
checks; `cd web && npm run ci` (258 vitest tests plus the build);
`examples/host_digest.py` re-run and read in all five states.

Pending user checks (not resolved by this review):

- manual: after a week of daily digests, the operator still reads them. It needs a
  week and the deployment. `examples/host_digest.py` prints all five states and is the
  fastest way to judge the wording now - and R1.1 is the finding most likely to have
  decided that question, which is why it was worth a round.
