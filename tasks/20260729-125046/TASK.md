# Add scheduled host checks and a proactive digest

- STATUS: OPEN
- PRIORITY: 40
- TAGS: feature,v0.1.0,host,telegram,backend

## Story

As the operator, I want Scufris to tell me things before I ask, so that it feels
like an assistant watching the machine rather than a chat window that happens to
know about it.

Nothing in Scufris is currently proactive: every run starts from the UI, a chat,
or Telegram. `wake.py` wakes the ORCHESTRATOR when a sub-agent needs it; nothing
wakes the OPERATOR. This task adds the smallest honest version of that: a
scheduler, a set of host checks, and a digest that arrives on its own.

## Steps

- [ ] Add the trigger mechanism: named schedules (interval or time-of-day)
      persisted with runtime state, surviving restart, with a recorded last-run
      and last-result, and no overlapping run of the same schedule.
- [ ] Add the check set, each a bounded read with an explicit threshold: disk
      pressure, failed systemd units, thermal/battery health, Nix store growth
      and garbage-collectable space, flake input staleness, and Scufris's own
      health (backend reachable, sessions loadable, state consistent).
- [ ] Add the digest: one message that reports what changed and what needs
      attention, with the boring case being short. A digest with nothing to say
      must say so briefly or not send at all - decide which, and be consistent.
- [ ] Deliver through Telegram (already the operator's off-desk surface) and
      keep the last digests readable in the dashboard.
- [ ] Let a check escalate: a threshold breach may propose a host action (for
      example garbage collection when the store crosses its limit), which enters
      the normal approval queue rather than acting on its own.
- [ ] Make it controllable: enable/disable per schedule, run-now, mute window,
      and configuration from settings rather than code edits.
- [ ] Cover the failure paths: a check that raises, a check that hangs, Telegram
      unreachable, and a schedule that fires while the previous run is still
      going.

## Definition of Done

- A schedule fires, the checks run, and the digest is delivered without any
  operator interaction (test: `test_scheduled_host_digest_is_delivered`).
- Schedules and their last-run state survive a restart, and a missed window does
  not stampede on startup (test: `test_schedules_survive_restart_without_stampede`).
- A failing or hanging check degrades the digest with a named failure instead of
  suppressing the whole message (test: `test_digest_survives_a_failing_check`).
- A threshold breach that proposes an action goes through the approval queue and
  never self-applies (test: `test_check_escalation_requires_approval`).
- manual: after a week of daily digests, the operator still reads them.

## Notes

- Epic: 20260729-124655.
- Depends on: the read-only inspection tools, and the action framework for
  escalation.
- KISS: this is a scheduler for HOST CHECKS, its first consumer. Generalizing it
  into arbitrary scheduled agent runs ("every morning, review my repos") is real
  and wanted, but it belongs in a later release once this one has proven the
  mechanism.
- Digest quality is the whole point. A noisy digest gets muted, and a muted
  digest is a deleted feature.

## Flow State

- FLOW STEP: PLANNING
