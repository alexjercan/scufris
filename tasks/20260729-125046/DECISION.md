# Decision: the digest is code-rendered, silent unless it has news, and never acts

- DATE: 20260730-125046
- STATUS: ACCEPTED
- TASK: 20260729-125046
- TAGS: decision, host, telegram, scheduler, v0.2.0

## Context

This is the epic's last child and its only PROACTIVE surface: everything Scufris
does today starts from the UI, a chat, or Telegram. `wake.py` wakes the
ORCHESTRATOR when a sub-agent needs it; nothing wakes the OPERATOR.

What already exists and is therefore not in question:

- every fact a check needs is a bounded read on `host.HostInspector` (units,
  journal, storage, generations, reclaimable space, thermals, flake status) plus
  `health.agent_health` for Scufris's own state;
- `Supervisor` runs and bounds background work, and the settings store
  (`settings_store.py`) already gives runtime-editable, persisted, whitelisted
  config with a dashboard surface;
- a proposal reaching the operator is solved twice over: the `/host/` queue and the
  Telegram surface, both behind one `HostApprovalService`.

Four forks were put to the operator before planning, because each one changes what
gets built rather than how.

## Decision

### 1. The checks are CODE, and so is the digest text

Each check is a python function with an explicit threshold, and the digest is
rendered from their results. No model turn is involved.

Rejected: letting the host agent write the digest (either as prose over the facts,
or by running it as a scheduled agent turn). The reasons, in order of weight:

- the DoD's own tests need thresholds and named failures - "a failing check degrades
  the digest with a named failure" and "a breach proposes an action" are not
  properties a prose turn can be tested for;
- the boring case must be SHORT, and that is a guarantee code gives and a model
  approximates;
- a digest is only worth reading if it is trustworthy, and non-determinism is the
  wrong thing to spend the first version's credibility on;
- the task's own note defers scheduled AGENT runs to a later release, and a
  prose-writing turn is the same mechanism by another name.

The prose layer stays available later without touching the check set: the checks
return structured results, so a renderer that asks the agent to phrase them is an
addition rather than a rewrite.

### 2. Silence unless there is news, plus one daily all-clear

Two schedules ship, and they differ in exactly this:

- `watch` (interval, default 15 minutes): sends only when something CHANGED - a check
  that entered or worsened into an attention state, or recovered. Nothing to say means
  nothing is sent.

  AMENDED DURING THE BUILD. This first read "sends when a check IS in a warn/crit
  state", which was implemented literally and measured at 96 messages a day for a disk
  that had not moved: a standing condition is not news, and re-sending it every
  interval is exactly how the feature gets muted (review round 1, R1.1). A condition
  the operator has already been told about is repeated by the DAILY line, which is one
  of the things that line is for.
- `daily` (time of day, default 08:00): always sends, even if it is one line.

So silence never means "is it even running?" - the heartbeat is the daily line - and
the noise floor is one message a day. The dashboard also shows the last run and its
result for both schedules, which is the answer to "did the 15-minute one fire".

Rejected: always sending every run (the shape most likely to be muted, and the task
says a muted digest is a deleted feature) and never sending an all-clear (a broken
scheduler would then look exactly like a healthy machine).

### 3. An in-process loop, with its state persisted

One asyncio loop in the app compares each schedule's persisted `next_due`;
`last_run`, `last_result` and `last_digest_at` survive a restart.

A window missed while the app was down is RECORDED as missed and skipped - it does
not fire on startup, and N missed windows do not stampede into N runs. The daily
schedule catches up at most once (a digest is about the machine now, not about a
morning that has passed).

Rejected: systemd timers in `nix.dotfiles` calling an API endpoint. It is the
NixOS-native shape and would give real OnCalendar semantics, but the app has to be
up for the call to land either way - so it buys calendar semantics rather than
reliability, at the price of a config-repo change and a new credential for the timer
to authenticate with. Worth revisiting if scheduled AGENT runs arrive, where a
missed window matters more.

### 4. Escalation ships, and ships OFF

A threshold breach may propose a host action, which enters the normal approval queue
and is never self-applied. Two constraints are structural rather than configured:

- only the R2 disposable-cleanup verbs may ever be escalated (`gc_store` today),
  enforced by an allowlist in code with a test that nothing else can be proposed
  this way. A check may not escalate a unit restart or an activation - those are
  decisions with consequences a threshold cannot judge;
- escalation is per check and defaults to OFF, so the first weeks are digests only;
- and it asks ONCE. Amended during the build (review round 1, R1.2): a breach
  escalated on every run while it lasted, so a full store with escalation on meant a
  new pending root-action proposal every fifteen minutes. A check now escalates only
  when its state CHANGED into the breach, and never while an equivalent proposal from
  the checks is still decidable.

Rejected: on by default for the store-pressure check. It is the genuinely useful
case (a full store, and a ready-to-approve collection next to the digest that
explains it), but it means two notifications for one event and a pending approval the
operator did not ask for - and the whole task's risk is being muted. The operator
turns it on once the digests have earned their attention.

## Consequences

- New modules: `scufris/checks.py` (the check set + thresholds),
  `scufris/digest.py` (render + the bounded persisted store),
  `scufris/scheduler.py` (named schedules, the tick loop, run-now, mute).
- Every check runs off the event loop (`asyncio.to_thread`) with its own timeout -
  the ledger's `sync-read-inline-on-a-latency-loop-stalls-it`. A hung read cannot be
  cancelled once handed to a thread; the timeout bounds the DIGEST, and the
  inspector's own subprocess timeouts bound the thread.
- Delivery reuses the Telegram bot the app already starts, and a `GET
  /api/host/digests` read for the dashboard's `/host/` page.
- Config lives in `Settings` and is whitelisted for runtime edits in the settings
  store, so enable/disable, the interval, the time of day, the thresholds and the
  mute window are all editable without a restart or an env change.
- The epic's Done Means 5 is satisfied by the `watch`+`daily` pair; its `manual:`
  item ("after a week of daily digests, the operator still reads them") remains the
  real acceptance test and cannot be closed by this task.
