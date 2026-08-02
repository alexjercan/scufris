# Add scheduled host checks and a proactive digest

- PRIORITY: 40
- TAGS: feature, v0.2.0, host, telegram, backend
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As the operator, I want Scufris to tell me things before I ask, so that it feels
like an assistant watching the machine rather than a chat window that happens to
know about it.

Nothing in Scufris is currently proactive: every run starts from the UI, a chat,
or Telegram. `wake.py` wakes the ORCHESTRATOR when a sub-agent needs it; nothing
wakes the OPERATOR. This task adds the smallest honest version of that: a
scheduler, a set of host checks, and a digest that arrives on its own.

`DECISION.md` in this folder settles the four forks: the checks and the digest text
are CODE (no model turn), the `watch` schedule is silent unless it has news while
the `daily` one always sends, the trigger is an in-process loop with persisted
state, and escalation ships but ships OFF.

## Steps

- [x] Add `scufris/scheduler.py`: two named schedules - `watch` (interval) and
      `daily` (time of day) - each with persisted `next_due`, `last_run`,
      `last_result` and `last_digest_at` under the state dir. No overlapping run
      of the same schedule (a fire while the previous run is in flight is recorded
      as skipped), and a window missed while the app was down is recorded as
      missed rather than fired - N missed windows must not become N runs.
- [x] Add `scufris/checks.py`: the check set, each a bounded read with an explicit
      threshold from settings, returning a structured result (name, state
      ok|warn|crit|unavailable|failed, headline, detail lines, and the facts it
      judged). The set: disk pressure, failed systemd units (both scopes),
      thermal health (this host is a DESKTOP - coretemp plus the CPU's
      thermal_throttle counters, no battery or fan), Nix store growth and
      collectable space, flake input staleness, and Scufris's own health (backend
      reachable, sessions loadable, state dir writable).
- [x] Run every check OFF the event loop with its own timeout
      (`sync-read-inline-on-a-latency-loop-stalls-it`), and say plainly in the
      code that a thread handed a hung read cannot be cancelled - the timeout
      bounds the DIGEST, and the inspector's own subprocess timeouts bound the
      thread.
- [x] Add `scufris/digest.py`: render the results into one message that leads with
      what needs attention, states what CHANGED since the last digest, and keeps
      the boring case to one line. Plus a bounded, persisted store of the recent
      digests so a restart does not lose yesterday's.
- [x] Deliver through Telegram (the operator's off-desk surface): the `watch`
      digest only when a check is warn/crit or has recovered, the `daily` one
      always. A Telegram failure must not lose the digest - it is stored and
      readable either way, and the schedule's `last_result` says the delivery
      failed.
- [x] Show the recent digests on the `/host/` page, with each schedule's last run
      and result, so "did the 15-minute check fire" has an answer that is not
      silence.
- [x] Let a check escalate: a breach may propose a host action, which enters the
      normal approval queue and is never self-applied. Only the R2 disposable
      cleanup verbs may ever be escalated (`gc_store` today), enforced by an
      allowlist in code, and escalation defaults to OFF per check.
- [x] Make it controllable from settings rather than code edits: enable/disable
      per schedule, the interval, the time of day, each threshold, a mute window,
      and run-now - all whitelisted in the settings store so they are editable at
      runtime and survive a restart.
- [x] Cover the failure paths: a check that raises, a check that hangs, Telegram
      unreachable, and a schedule that fires while the previous run is still
      going.
- [x] Update the docs surfaces in THIS task: README (what arrives, when, and how
      to mute it), AGENTS.md (the scheduler's shape and the escalation
      allowlist), CHANGELOG, and an `examples/host_digest.py` that prints a digest
      for each state - all clear, a warn, a crit with an escalation, and one with
      a failed check.

## Definition of Done

- A schedule fires, the checks run, and the digest is delivered without any
  operator interaction (test: `test_scheduled_host_digest_is_delivered`).
- Schedules and their last-run state survive a restart, and a missed window does
  not stampede on startup (test: `test_schedules_survive_restart_without_stampede`).
- A failing or hanging check degrades the digest with a named failure instead of
  suppressing the whole message (test: `test_digest_survives_a_failing_check`).
- A threshold breach that proposes an action goes through the approval queue and
  never self-applies, and only an R2 verb can be proposed this way
  (test: `test_check_escalation_requires_approval`).
- The boring case is silent on `watch` and one line on `daily`, and a muted window
  suppresses delivery while still recording the run
  (test: `test_the_boring_case_is_silent_except_for_the_daily_line`).
- A Telegram failure does not lose the digest: it is still stored, still readable,
  and the schedule records the failed delivery
  (test: `test_a_delivery_failure_keeps_the_digest`).
- The dashboard shows the recent digests and each schedule's last run
  (test: `renderHostDigests` vitest suite).
- cmd: `python -m pytest`
- cmd: `nix flake check`
- cmd: `cd web && npm run ci`
- manual: after a week of daily digests, the operator still reads them.

## Notes

- Epic: 20260729-124655. This is its last child; Done Means 5 is what it closes.
- Depends on: the read-only inspection tools (20260729-125024, landed), the action
  framework for escalation (20260729-125029, landed), and the Telegram surface
  (20260730-104524, landed) whose bot this reuses for delivery.
- `DECISION.md` (this folder): the four forks and what was rejected.
- KISS: this is a scheduler for HOST CHECKS, its first consumer. Generalizing it
  into arbitrary scheduled agent runs ("every morning, review my repos") is real
  and wanted, but it belongs in a later release once this one has proven the
  mechanism.
- Digest quality is the whole point. A noisy digest gets muted, and a muted
  digest is a deleted feature.
- Two schedules with FIXED identities (`watch`, `daily`) rather than
  operator-defined ones: the config then fits in plain settings fields instead of a
  parsed schedule language, which is the KISS note applied to the trigger.
- Ledger lessons that bite here: off-load synchronous reads in a loop
  (`sync-read-inline-on-a-latency-loop-stalls-it`), cap what must SURVIVE a trim
  rather than the length (`cap-what-must-survive-not-just-the-length` - a digest is
  a rendered document with a lead), and a test that asserts "nothing was sent"
  needs a paired delivery guard proving the run actually fired.
