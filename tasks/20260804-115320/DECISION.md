# Decision: the provider session is a cache, and assembly is a window

- DATE: 20260804-115320
- STATUS: ACCEPTED
- TASK: 20260804-115320
- TAGS: v0.2.0, lane1, chat, context, cache

## Context

`tasks/20260729-220835/DECISION.md` section 1 makes the semantic conversation
the source of truth and the provider session a cache keyed by
`(conversation, backend, policy version)`, re-seeded "from assembled context"
when the binding is invalid. It fixes the key and the direction and nothing
else.

Four questions were left for whoever built it, and `tasks/20260804-115320/NOTES.md`
settled one of them (summarization is cut; the bound is a window) with its
evidence. This record ratifies that cut, answers the other three, and records
each rejection with the trigger that would reopen it.

## Decision

### 1. Keyed for LOOKUP by the triple, PRIMARY-KEYED by `(conversation, backend)`

`provider_session` is `PRIMARY KEY (conversation_id, backend)`.
`policy_version` is a constrained column the read must MATCH, not a third key
column.

The ratified key is a lookup key and stays one: `cached_session` returns `None`
for an absent row, for a row under a different `policy_version`, and for a
conversation that does not exist. What changes is what happens to the row a
miss replaces.

With the triple as the primary key, one binding per policy version accumulates
and a policy DOWNGRADE resurrects a dead one. A row seeded under version 1,
superseded by version 2, has since missed every event appended under version 2
- it is bound to a provider session that stopped at the old event log. Going
back to version 1 finds it and reads it as WARM. Nothing detects that; the
operator gets a session that has forgotten an hour of conversation and no error
anywhere.

One live binding per `(conversation, backend)` makes a re-seed an UPSERT that
OVERWRITES, so the stale row cannot come back and nothing accumulates. A
downgrade then misses and re-seeds, which is the correct and only safe answer.

### 2. Summarization is deferred, not missing

The bound on assembled context is a WINDOW over the most recent events. There
is no summarizer in v0.2.0. `NOTES.md` holds the four-item evidence: the
accepted decision never asks for one; the thing being generalized
(`format_fork_seed`, `scufris/sessions/transcript.py:177`) already bounds by
windowing; a Scufris-side summarizer calls a model, which the offline example
this lane is proven by cannot do; and a summarizer needs a compactor, storage,
an invalidation rule and an answer to who writes it.

The accepted cost: the provider stops seeing the early part of a long
conversation. Nothing is deleted - the semantic log is intact and the operator
reads all of it - so the release promise holds as written. But "the model forgot
what we discussed an hour ago" is a real experience.

**Trigger to reopen:** the first time a window drops context the operator
actually needed.

Summary VERSIONING falls away with it: there is no summary to version.
`policy_version` survives for what `NOTES.md` says it is for - the
system/project policy and the presets legal right now, which change
independently of any summary.

### 3. Lazy re-seed. A backend switch writes nothing at all

A switch to backend B is "use B next turn". It writes no row, seeds no session
and does no work. The next turn's `cached_session` misses, the caller assembles
and binds, and that is the whole path.

Eager re-seeding at switch time was considered and rejected. The argument for it
is real - it pays the assembly cost at a moment the operator is already waiting
- but nothing in v0.2.0 requires it. The lazy path is the whole path; eager
would be a SECOND way to reach one state, both to be kept correct, for a mode
with no caller.

This rejection does NOT rest on "a restart and a compaction force the lazy path
anyway". They do not reach it: the row is durable, so a restart reads it back
warm, and nothing reports a provider-side compaction, so `cached_session` misses
on neither. Both are undetected staleness, recorded under Paid below.

**Trigger to reopen:** a measurable, operator-visible stall on the first turn
after a switch.

### 4. The bound is an EVENT COUNT, and that is a proxy

`CONTEXT_WINDOW_EVENTS` counts events, not characters and not tokens. It is
honestly a proxy: one enormous body still overflows a provider that a hundred
small ones would not.

A character or token bound is deferred rather than added alongside. Two knobs
before either has a real caller is one knob too many, and the second one cannot
be tuned by anybody who has not yet watched the first one fail.

**Trigger to reopen:** the first time a windowed assembly still overflows a
provider.

### 5. Assembled context is attributed, and only the operator instructs

Every line of the assembled prompt carries its actor - `operator`,
`agent:<id>`, `orchestrator`, `system` - and the preamble states that only the
operator's lines are instructions.

This is `tasks/20260729-220835/DECISION.md` section 3 one layer down. An agent
report is an untrusted QUOTATION; flattening it into the same undifferentiated
prose as the operator's messages hands a provider a prompt in which an agent's
output is indistinguishable from an instruction, which is the exact defect
`20260804-115321` exists to fix at the message-role layer. Assembly is where it
would be reintroduced, so the format is fixed here and asserted there.

## Alternatives considered

- **The triple as the primary key.** Rejected in section 1: bindings accumulate
  per policy version and a downgrade resurrects one that has missed every event
  appended since, read as warm with no error anywhere.
- **A summarizer instead of a window.** Rejected in section 2 with the four
  items `NOTES.md` records; deferred with its reopening trigger, not dropped.
- **Eager re-seed on a backend switch.** Rejected in section 3: a second path to
  a state the lazy path already reaches, for a mode v0.2.0 does not require.
- **A character or token bound, now or alongside the event count.** Rejected in
  section 4: two knobs before either has a caller.
- **An unattributed transcript, or one that renders every line as the
  operator's.** Rejected in section 5: it reintroduces at the prompt layer the
  defect `20260804-115321` fixes at the role layer.
- **A `forget_session` invalidation function.** Not built: `/new` mints a new
  `conversation_id` (`tasks/20260729-220835/DECISION.md` section 6), so the old
  binding is never looked up again rather than dropped. It would be a function
  with no caller.
- **A `seeded_through_seq` column.** Not built: events appended while a binding
  is warm - an agent report landing between turns - are not in the provider's
  working memory, and nothing in v0.2.0 detects that. Reopening trigger: the
  first caller that appends a non-operator event outside a turn it is itself
  driving.

## Consequences

**Gained.** The conversation survives `/new`, compaction, a backend switch and a
restart, because the provider session is now a cache row that can be thrown away
and rebuilt from the event log. A cache miss is normal and raises nowhere.
Assembly is bounded IN SQL - `ORDER BY event_seq DESC LIMIT n`, then reversed -
so a bounded result costs bounded work rather than reading the whole
conversation and slicing it.

**Paid.** A long conversation's early events stop reaching the provider (section
2). The bound is a proxy and can be wrong in the large-body direction (section
4). The first turn after a backend switch pays the assembly and seeding cost
(section 3).

And three kinds of UNDETECTED staleness, all the same shape - the row reads warm
and the provider session behind it is not what it claims:

- A **restart**. The row is durable, so the binding comes back warm and points
  at a provider session the provider may no longer hold. Reopening trigger: a
  resume against a dropped session is seen to fail, or to answer from an empty
  history.
- A provider-side **compaction**. Nothing reports it, so `cached_session` cannot
  miss on it. Same trigger.
- Events appended while a binding is warm, outside a turn the appender is
  driving. This is the deferred `seeded_through_seq`; its trigger is the first
  caller that does it.

None of the three is a cache miss, which is the correction section 3 above now
carries: `cached_session` misses on an absent row or a policy mismatch, and on
nothing else.

**Not addressed here.** The turn-driving code that calls these functions (a
later lane), summarization, retention, and the colour-per-actor rich transcript
the epic's Lane 1 blurb mentions - that is a rendering concern for Lane 8, not
part of the assembled prompt fixed here.

**Reversal.** Dropping `provider_session` and letting the provider's own session
be the conversation is exactly the pre-v0.2.0 behaviour this replaces, and it is
what makes `/new`, a switch and a restart lose the thread.
