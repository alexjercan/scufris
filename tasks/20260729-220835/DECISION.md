# Decision: the actor-aware conversation, its four records, and flow authority

- DATE: 20260803-184746
- STATUS: ACCEPTED
- TASK: 20260729-220835
- TAGS: v0.2.0, v0.3.0, agents, orchestrator, projects, telegram, architecture

## Context

`tasks/20260801-100405/DECISION.md` closed the persistence question and
deliberately left two answers to this task: what semantic record Scufris keeps
above provider-owned native transcripts (its section 3), and the conversation,
activity and delivery tables its constraint 5 reserved but did not create.

`tasks/20260729-220835/SPIKE.md` located the defects in the shipping code and
compared four ownership models against them. This record is what the v0.3.0
implementation epic must honor. It creates no schema and changes no behavior.

This record is ratified at the task's manual acceptance round - the operator
playing through `tasks/20260729-220835/mockup.html` - and is amended before the
spike closes if that round redirects it. `tatr` recognizes no PROPOSED status,
so the header says ACCEPTED and this paragraph is where the gate lives.

## Decision

### 1. Scufris owns a SEMANTIC conversation; the provider session is a cache

The product conversation is a Scufris-owned, ordered, actor-attributed log of
what was said and decided. It is not a transcript copy: it records semantic
events and links to the native transcript for detail.

The provider session keeps the model's working memory and its native tool and
reasoning fidelity, and is bound to a conversation as a CACHE keyed by
`(conversation, backend, policy version)`. A warm, valid binding is resumed and
appended to, which is today's behavior and preserves the prompt cache. An
invalid or absent binding is re-seeded from assembled context. Invalidating a
binding costs one re-seed and never costs the conversation, which is what makes
a `/new`, a compaction, a backend switch and a restart non-destructive.

Owning the FULL provider transcript stays rejected on the evidence in
`tasks/20260724-111839/SPIKE.md`; nothing has changed to reopen it.

### 2. Four records, four owners, and no projection becomes a source

| Record | Owner | Authoritative for |
|-|-|-|
| Semantic conversation | Scufris database | what the operator and the system said and decided |
| Technical activity | Scufris database | what a run did - tools, phases, exit, worktree |
| Provider transcript | the provider CLI | the model's working memory and native fidelity |
| Enforcement audit | `hostd` (root) and the tatr files | privileged actions applied; task lifecycle truth |

The conversation stores no lifecycle truth, the activity log stores nothing that
was said, the audit is never app-writable, and no surface reads two of these for
the same fact. Every operator view is a PROJECTION of exactly one of them.

### 3. Actor attribution is typed, and only the operator may authorize

Every semantic event carries an actor: `operator` (with the channel it arrived
on), `orchestrator`, `agent:<id>` (with its preset and run), or `system`.

**An agent report is data, never an instruction.** It enters assembled context
as an attributed, untrusted quotation, and only an `operator` event may satisfy
a stop gate. This is what makes today's `wake_prompt` - which is recorded as a
`user_message` and re-rendered in the operator's own voice
(`scufris/wake.py:44`, `scufris/sessions/transcript.py:94`) - a defect with a
structural fix rather than a cosmetic one.

### 4. Correlation and idempotency invariants

- `event_seq` is monotonic per conversation and assigned inside the writing
  transaction, as `HostActionRow.seq` and `ConfigChangeRow.seq` already are.
- `correlation_id` is one per operator intent; every event that intent causes
  carries it. `causation_id` names the event that directly caused this one.
- Delivery is keyed `PRIMARY KEY (channel, idempotency_key)` with
  `idempotency_key = (conversation_id, event_seq)`. A redelivery after restart
  is a no-op. This replaces `TelegramApprovals._announced`, an in-memory capped
  `OrderedDict` (`scufris/telegram/approvals.py:78`).
- A run joins the conversation by `run_id -> correlation_id`, so the activity
  timeline and the conversation are two views of one tree.
- The state change and its event commit in ONE transaction, per
  `tasks/20260801-100405/DECISION.md`.

### 5. Workflow authority stays with tatr; Scufris asks and renders

Scufris stores assignments and observations, never lifecycle truth. Before every
launch or transition, one server-side guard: re-read the task through the typed
reader (20260729-102158); probe legality with `tatr flow -n <id>` and keep its
named preconditions; check no conflicting active run through the existing
`AgentRunService.launch` claim; require an `operator` approval event for this
transition's stop gate; and on refusal return the REASON, which the UI renders
instead of an unexplained disabled control.

The flow state machine Scufris draws is a projection for the operator to read,
not a second engine.

### 6. Channels are two projections of one conversation

One conversation, ordered replay by `event_seq`, deduplicated delivery. Telegram
receives decisions and outcomes - approval requests, gate results, final
answers, failures - not reasoning deltas or tool widgets, which stay a nicety of
the turn that surface itself started. "New chat" mints a new `conversation_id`
on both surfaces and destroys nothing. An approval decided from either channel
writes ONE decision event and the other channel's card resolves on replay. A
pending stop gate is reconstructed from the log after a restart.

### 7. Partial supersession of "orchestration pipelines are dropped"

`tasks/20260720-184150/SPIKE.md` Revision 1 dropped Option D, orchestration
pipelines. That rejection STANDS for what it named: a generic workflow engine
over agents, invented by Scufris, with no authority behind its states.

It is superseded only for the project flow coordinator recommended here, which
is not that: it coordinates over an external state machine tatr already owns and
already enforces, and its whole job is to project legal moves, hold the four
operator stop gates, and record who authorized each one. A pipeline would
decide; this asks and renders.

## Alternatives considered

Full reasoning in `tasks/20260729-220835/SPIKE.md`.

- **Provider session as the product conversation (status quo).** Rejected on six
  located defects: no actor, so a machine prompt renders as the operator; no
  cross-channel delivery; no Telegram replay; a `/new` that silently destroys
  the other surface's history; per-backend transcript fidelity; and a backend
  switch that erases the conversation by construction. It also has nowhere to
  record an approval, a delivery, or a transition.
- **Scufris owns the full provider transcript.** Rejected in
  `tasks/20260724-111839/SPIKE.md` on fidelity and prompt caching; re-checked
  and not reopened.
- **Patch the symptoms** - mark the wake prompt, add a Telegram broadcast hook,
  persist `_announced`. Each is small and lands quickly, and none leaves a
  record that can answer why an action is illegal, who approved a gate, or what
  a channel has already been told.

## Consequences

**Gained.** "Who said this" and "may this authorize a gate" become typed
queries. The conversation survives reset, fork, compaction, backend switch and
restart. Delivery becomes durable and idempotent, so the browser and the phone
are two projections of one ordered log rather than two surfaces that drifted. An
unavailable action carries a reason. The four stop gates become reconstructible
after a restart, which they are not today.

**Paid.** A second write on every turn, inside the same transaction as the state
change. A real risk of the semantic log drifting from the provider transcript,
bounded by the log deliberately not being a transcript. New tables under a
retention policy that does not exist yet. Context assembly becomes code Scufris
owns and must keep bounded, where today the provider does it.

**Not addressed here.** Retention policy, summary versioning, per-turn event
granularity, eager-versus-lazy re-seed on a backend switch, the
`SCUFRIS_ORCH_SESSION_ID` rename, and where the guard service lives - all listed
as open questions in `SPIKE.md` and scoped to v0.3.0 tasks.

**Reversal.** The semantic log is additive: provider sessions keep working
exactly as they do today, so removing the layer means dropping tables and
falling back to `read_transcript`, at the cost of the conversation history it
held.
