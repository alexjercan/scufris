# Spike: define the actor-aware orchestrator conversation and flow-control model

- DATE: 20260803-184035
- STATUS: RECOMMENDED
- TAGS: spike, v0.2.0, agents, orchestrator, projects, telegram, frontend

## Question

Who owns the product conversation, and what is authoritative for workflow
state, given that provider CLIs own rich native transcripts, two channels (web
and Telegram) drive one orchestrator, tatr files own lifecycle truth, and the
app now has one transactional database?

Answered concretely enough that the v0.3.0 implementation epic
(`tasks/20260801-154211/TASK.md`, "Epic 2: Actor-aware project coordination")
can be seeded without re-litigating conversation ownership, actor attribution,
provider-session use, context assembly, channel delivery, workflow authority,
or recovery.

Out of scope: production schema, production UI, and the implementation tasks
themselves. The artifact that makes the answer concrete is
`tasks/20260729-220835/mockup.html`, played through end to end.

## Context

Grounded in a read of the shipping code after the host epic (20260729-124655)
and the durability epic (20260729-102145, lanes A/B landed). The load-bearing
facts, each with its location:

### The conversation is a provider session, and it cannot say who spoke

- **One reserved orchestrator, one current session.** `agent_session` holds
  `current_session_id` + `backend` per agent (`scufris/db/models.py:92`), and
  `OrchestratorTurnService` is the single turn path for all three transports -
  the landing chat, the Telegram bot and the wake bridge
  (`scufris/orchestrator/turn.py:28`). "Web and Telegram share a conversation"
  is already true in the sense that they share a SESSION.
- **The web renders history by re-reading the provider store.**
  `read_transcript` folds a codex rollout into `TranscriptMessage(role=...)`
  (`scufris/sessions/transcript.py:61`), and the role vocabulary is exactly
  `user` and `assistant` (`scufris/sessions/models.py`). There is no actor.
- **So a machine-authored prompt renders as the operator.** `wake_prompt`
  builds `"[wake] One or more sub-agents need your attention: ..."`
  (`scufris/wake.py:44`); `OrchestratorTurnService.wake` launches it as the
  turn's message (`scufris/orchestrator/turn.py:98`); codex records it as a
  `user_message`; `read_transcript` strips only the sentinel-wrapped steering
  block (`scufris/sessions/steering.py:20`, "removes only the single leading
  block") and emits `role="user"`. The operator reloads the dashboard and reads
  a message they never sent, in their own voice. This is the defect that makes
  actor attribution structural rather than cosmetic.
- **An agent's report arrives the same way.** `report_back` and `request_input`
  reach the orchestrator as text inside a wake prompt or a `pending_agents`
  read. Nothing in the record distinguishes "a sub-agent claims the work is
  done" from "the operator said ship it", which is the exact confusion a
  workflow gate must not make.

### The two channels are not one conversation

- **Telegram renders only the turn it started.** `TelegramBot._drive_turn` is
  called from `poll_once` on an inbound update and consumes the stream from its
  own `on_message` (`scufris/telegram/bot.py:223,253`). A turn started from the
  web, or granted by the wake bridge, is never delivered to the phone.
- **Telegram has no replay.** `drive_turn` paints live and keeps nothing
  (`scufris/telegram/turn.py:81`). Reopening the chat shows whatever Telegram
  itself retained, in whatever order the surface happened to write it.
- **`/new` is a shared, silent reset.** `on_reset` calls `turn.reset()`, which
  clears the orchestrator's `current_session_id`
  (`scufris/orchestrator/turn.py:119`). A `/new` from the phone destroys the
  history the browser was reading, with no event either surface can show.
- **One push path exists, and it proves the shape is needed.** Host approvals
  ARE announced to Telegram (`host_approval_bridge._announce`
  -> `TelegramApprovals.announce_proposal`, `scufris/telegram/approvals.py:101`)
  and edited when decided. Its delivery bookkeeping is `self._announced`, an
  in-memory `OrderedDict` capped at `MAX_TRACKED_ACTIONS`
  (`scufris/telegram/approvals.py:78`): lost on restart, evicted under load. So
  the app already needs durable, deduplicated delivery and currently
  approximates it once, for one record type.

### Fidelity is per-backend and a session id is not portable

`read_transcript` is four different readers behind one protocol method: a codex
rollout file (`backends/codex.py:112`), a claude project JSONL
(`backends/claude.py:476`), an opencode HTTP call (`backends/opencode.py:248`),
and `[]` for mock (`backends/mock.py:51`). `agent_session.backend` exists
precisely so a stale cross-backend id is unreachable - "a codex rollout id means
nothing to claude, so every accessor matches on it and a backend switch starts a
fresh history" (`scufris/db/models.py:100`). Under today's model, switching
backend is indistinguishable from erasing the conversation.

### Workflow truth is in files, and tatr already refuses illegal moves

- Scufris reads tasks by shelling `tatr -r <root> ls` and regex-matching the
  display line into `{id, title, priority, tags}` (`scufris/projects.py:264`).
  ACTIVITY, GATES, RESOLUTION, review verdict, dependencies and sibling records
  are not read at all; 20260729-102158 is the task that fixes this with a typed
  reader.
- `tatr flow -n <id>` is a shipped legality probe: on this very task it refused
  with named preconditions (`bad-record-schema: TASK.md has no '## Question'
  section`, `missing-spike-record`) and wrote nothing. So Scufris does not have
  to re-implement flow legality - it can ASK, and render the refusal.
- The flow contract's four blocking stop gates - `NOTES_READY`, `PLAN_READY`,
  `WORK_DONE` (initial, and every third review round), `LAND_READY` - plus the
  direct `REQUEST_CHANGES -> work` and `APPROVE -> compound` routes are defined
  in `~/.claude/skills/flow/gates.md` and restated in
  `tasks/20260801-154211/TASK.md`.

### The persistence boundary is already decided, and it reserved this workload

`tasks/20260801-100405/DECISION.md` constraint 5 states it explicitly: no
conversation, activity-event or delivery tables are created by that epic, and
the chosen store carries them - "ordered append, correlation index,
`PRIMARY KEY (channel, idempotency_key)` for idempotent delivery, retention by
`DELETE`, and an atomic state-plus-event commit - through a normal migration
when 20260729-220835 designs them". Its section 3 also reserves the answer this
spike owes: provider-owned native session transcripts are outside the boundary,
and "20260729-220835 decides what semantic record Scufris keeps above them".

Two rules from that decision bind everything below: a transaction is the
read-modify-write boundary and never spans an `await`, and loop-thread callers
offload through `asyncio.to_thread`.

### Prior decisions this spike must reconcile

| Record | What it settled | Standing after this spike |
|-|-|-|
| `20260720-184150/SPIKE.md` | "multiple agents" = per-project agents; Option D **orchestration pipelines dropped** | SUPERSEDED in part - see Recommendation 6 |
| `20260720-221748/SPIKE.md` | an agent instance is a server-driven record, not a detached process; v1 observation is read-only | Kept. Instances stay records |
| `20260723-001256/SPIKE.md` | durable run outcome + `WAITING`, `request_input`, `pending_agents`, the wake bridge | Kept. The wake bridge becomes an event PRODUCER rather than a prompt author |
| `20260724-111839/SPIKE.md` | hybrid: provider-owned transcripts + a Scufris-owned ownership index; owning the full transcript REJECTED on fidelity and prompt caching | Kept, and extended - the index grows a semantic layer, not a transcript copy |
| `20260724-132713/DECISION.md` | escalations route by `SCUFRIS_ORCH_SESSION_ID` to the spawning chat | Kept mechanically; generalized to `conversation_id` |
| `20260727-022121/DECISION.md` | sub-agent steering is backend-agnostic, via the turn prompt | Kept. Assembled context rides the turn prompt for the same reason |

## Options considered

### A. Provider session IS the product conversation (status quo)

The web re-reads the rollout; Telegram paints live; `reset` clears the pointer.

- **For:** zero new storage; native tool/reasoning fidelity; prompt caching for
  free; already shipped.
- **Against, all located above:** no actor, so a wake prompt renders as the
  operator (`wake.py:44` -> `transcript.py:94`); no cross-channel delivery, so a
  web turn never reaches the phone (`bot.py:253`); no replay on Telegram; `/new`
  silently destroys the other surface's history (`turn.py:119`); fidelity and
  shape change with the backend (four `read_transcript` implementations); a
  backend switch erases the conversation by construction (`models.py:100`); and
  there is no record in which an approval, a delivery, or a workflow transition
  could be written at all. Every product behaviour this spike exists to enable
  would have to be bolted onto a store Scufris does not own.

### B. Scufris owns the full provider transcript, re-injected each turn

Generalize `format_fork_seed` (`sessions/transcript.py:166`) to every backend
and every turn.

- Already rejected with evidence in `20260724-111839/SPIKE.md`: lossy for
  stateful CLIs whose rollouts hold native tool-call and reasoning events with
  no faithful plain-text re-injection; breaks prompt caching, so input cost
  grows quadratically; reinvents the compaction the CLIs already do.
- **Nothing has changed to reopen it.** The CLIs are still stateful,
  resume-append still works, and the frameworks that own their transcripts do so
  because their APIs are stateless. Recorded as rejected, not re-argued.

### C. A Scufris-owned SEMANTIC conversation above provider-owned sessions (recommended)

Scufris owns an ordered, actor-attributed log of what the PRODUCT considers said
and decided: operator messages with the channel they arrived on, assistant
answers, agent reports, approvals asked and decided, workflow transitions,
delivery receipts. The provider session keeps the model's working memory and the
native technical transcript, and becomes a CACHE keyed by
`(conversation, backend, policy version)` rather than the source of truth.

- **For:** attribution becomes a typed field instead of a role string, so "who
  said this" and "may this authorize a gate" are answerable; the conversation
  survives reset, fork, compaction, backend switch and restart, because none of
  those touch it; delivery gets a durable idempotency key, so the phone and the
  browser can be two projections of one ordered log; the mediator becomes
  logically stateless, so its context is a deterministic function of durable
  state rather than of whatever the provider happens to still hold; and B's two
  decisive costs are avoided, because the provider session is still resumed and
  appended to in the normal case.
- **Against:** a second thing to write on every turn, and a real risk of the
  semantic log drifting from the provider transcript. The mitigation is that it
  is deliberately NOT a transcript - it records semantic events, links to the
  native transcript for detail, and never claims to reproduce it.

### D. Patch the symptoms

Prefix the wake prompt with a marker; add a Telegram broadcast hook; persist
`_announced`.

- **For:** each fix is small and lands this week.
- **Against:** every one of them writes product meaning into a store Scufris
  does not own and cannot query. There would still be no record that can answer
  "why is this action illegal", "who approved this gate", or "what has this
  channel already been told", which are the three questions the v0.3.0 workspace
  is made of. Recorded as the honest cheap option, and rejected on that.

## Recommendation

**Option C.** Scufris owns a semantic conversation; provider sessions stay
provider-owned and become a cache; tatr files stay authoritative for lifecycle;
the root helper's audit stays external. The load-bearing calls are recorded in
`tasks/20260729-220835/DECISION.md`; what follows is the reasoning behind them.

### 1. Four records, four owners, one direction of projection

| Record | Owner | Authoritative for | Where it is read |
|-|-|-|-|
| Semantic conversation | Scufris database | what the operator and the system said and decided | `/` chat, Telegram, the Project conversation tab |
| Technical activity | Scufris database | what a run DID - tools, phases, exit, worktree | Activity tab, agent detail, run timeline |
| Provider transcript | the provider CLI | the model's working memory and native fidelity | the Agents console, by deep link only |
| Enforcement audit | `hostd` (root) and the tatr files | privileged actions applied; task lifecycle truth | the host audit page; the Project lifecycle badges |

The invariant that makes four records safe: **a projection never becomes a
second source.** The conversation stores no lifecycle truth, the activity log
stores nothing that was said, the audit is never app-writable, and no surface
reads two of these for the same fact.

### 2. Actors, and the one that may authorize

Every semantic event carries a typed actor:

| Actor | Example | May authorize a gate |
|-|-|-|
| `operator` (+ channel: web session, or telegram chat id) | "approve the plan" | **yes** |
| `orchestrator` | the mediator's own answer | no |
| `agent:<id>` (+ preset, run) | a planner's report, a reviewer's verdict | no |
| `system` | a wake, a scheduler pass, a host approval bridge event | no |

**An agent report is data, never an instruction.** When it enters the
orchestrator's assembled context it is wrapped as an attributed, untrusted
quotation. Only an `operator` event may satisfy a stop gate. That single rule
fixes today's wake-prompt defect, closes the "the reviewer said APPROVE so we
landed" hole, and is what the word "actor-aware" is actually buying.

### 3. Correlation and idempotency

- `conversation_id` + `event_seq`, monotonic per conversation, assigned inside
  the writing transaction - the pattern `HostActionRow.seq` and
  `ConfigChangeRow.seq` already use (`db/models.py:287,323`) and for the same
  reason: order is what the operator reads, and rowid is not promised.
- `correlation_id`, one per operator intent. Every event that intent causes -
  the turn, its activity, its agent runs, its approvals - carries it. This is
  what makes "which five things came from my message" a query.
- `causation_id`, the event that directly caused this one. Correlation gives the
  tree; causation gives the edges.
- Delivery keyed `PRIMARY KEY (channel, idempotency_key)` with
  `idempotency_key = (conversation_id, event_seq)` - the constraint
  `20260801-100405/DECISION.md` constraint 5 already reserved. A redelivery
  after restart is a no-op instead of a second buzz on the phone, which is what
  `_announced` cannot promise today.
- Runs join by `run_id -> correlation_id`, so the activity timeline and the
  conversation are two views of one tree rather than two logs to eyeball.

### 4. The provider session as a cache

Each orchestrator turn's context is assembled from durable state:

1. system and project policy (steering preambles, the capability list);
2. a versioned rolling summary covering events up to `seq N`;
3. semantic events after `N`, each actor-tagged;
4. pending decisions and workflow assignments;
5. the presets and capabilities that are legal RIGHT NOW.

Then: if a warm provider session exists and its binding is still valid - same
backend, same policy version - **resume and append only the new events**, which
preserves the prompt cache and is exactly today's behaviour. Otherwise **mint a
fresh session seeded with the assembled context**, which is `format_fork_seed`
generalized.

So the binding is a cache keyed by `(conversation, backend, policy version)`.
Invalidating it costs one re-seed and never costs the conversation. That is what
makes a backend switch, a compaction, a `/new`, and a restart non-destructive -
the four things today's model cannot survive - while keeping B's costs off the
normal path, because the normal path is still resume-append.

### 5. Workflow authority: ask tatr, do not mirror it

Scufris stores **assignments and observations**, never lifecycle truth. Before
every launch or transition the server runs one guard:

1. re-read the task's authoritative record through the typed reader
   (20260729-102158) - never a cached assignment;
2. probe legality with `tatr flow -n <id>` and keep its named preconditions;
3. check no conflicting active run, through the existing
   `AgentRunService.launch` claim (`orchestrator/runs.py`);
4. require an **`operator`** approval event for this transition's stop gate, in
   this conversation - an agent's claim never satisfies it;
5. on refusal, return the REASON, which the UI renders in place of an
   unexplained disabled control.

Step 2 is the point: `tatr flow -n` already refuses with named preconditions and
writes nothing, so "tatr files remain authoritative" becomes a mechanism rather
than a slogan, and the flow state machine cannot drift from the tool that owns
it. The state machine Scufris draws - `UNDERSTANDING -> PLANNING -> WORKING ->
REVIEWING -> COMPOUNDING -> DONE -> landed`, with `PLAN`/`REVIEW`/`RETRO`
earned, `REQUEST_CHANGES` routing directly back to work and `APPROVE` directly
to compound - is a PROJECTION for the operator to read, not a second engine.

### 6. This supersedes "orchestration pipelines are dropped"

`20260720-184150/SPIKE.md` dropped Option D, orchestration pipelines, and that
was right for what it was: a generic workflow engine over agents, invented by
Scufris, with no authority behind its states. What is recommended here is not
that. It is a **coordinator over an external state machine tatr already owns and
already enforces**, whose whole job is to project legal moves, hold the four
operator stop gates, and record who authorized each one. The distinction is the
authority: a pipeline would decide; this asks and renders. Recorded as a partial
supersession in `DECISION.md`, with the generic-pipeline rejection intact.

### 7. Channel semantics

- One conversation, two projections. Both read the semantic log; neither owns
  history.
- Ordered replay by `event_seq`. Telegram catch-up is bounded: the last N events
  plus a line saying how many more are in the dashboard.
- Delivery deduplicated by `(channel, idempotency_key)`; edits and batching are
  delivery-layer concerns and never rewrite an event.
- **Noise policy:** Telegram receives decisions and outcomes - approval
  requests, gate results, final answers, failures. Reasoning deltas and tool
  widgets stay a nicety of the turn that surface itself started. A web-driven
  turn reaches the phone as its answer plus any approval it raised, not as a
  transcript.
- "New chat" mints a new `conversation_id`. It is the same operation on both
  surfaces, both see it, and it does not destroy anything - the previous
  conversation is still readable.
- An approval is an event with typed choices. Deciding from either channel
  writes ONE decision event; the other channel's pending card resolves on
  replay. This is `host_approval_bridge`'s existing shape, generalized and made
  durable.
- Recovery: a pending stop gate is reconstructed from the event log, not from
  process memory. Today it is not durable at all.

### 8. Presets, instances, runs

`preset` (reusable: role, backend, model, permission mode, steering, capability
set) resolves at launch into an `agent instance` (reusable or ephemeral), which
owns `run`s, each binding a worktree, branch, commits and one provider session.
A stage assignment binds `(project, task, stage) -> preset`. The reserved agents
- the orchestrator and the host agent - are visible in the Agents console but
are never offered as stage presets. The preset schema itself is
20260729-102205/102206's to decide; this spike only fixes the relationships it
must satisfy.

## The mockup

`tasks/20260729-220835/mockup.html` - a single self-contained static file with
fixture data and no production integration, as agreed in the task Notes.

**Run it:**

```sh
xdg-open tasks/20260729-220835/mockup.html
```

No build, no server, no network. Everything is inline; it opens from `file://`.

**What it demonstrates.** One scenario, stepped forward by a single control, in
three linked views rendered together: the `/` conversation, the Project
workspace, and a phone-width Telegram projection of the SAME conversation. The
steps are the acceptance journey from `tasks/20260801-154211/TASK.md`: no task
-> planning agent -> PLAN_READY gate -> work -> WIP -> report -> review ->
changes requested -> fix -> approve -> compound -> LAND_READY. Every message
carries its actor and channel; the Project view carries lifecycle badges, the
assigned agent, the active run, branch and artifact links, and the legal next
action with a REASON on every unavailable one; the Telegram column shows only
what the noise policy says it should.

**Observations from playing it through** are recorded in the acceptance round
with the user and folded back into "Open questions", rather than paraphrased
here.

**Limitations, stated while it is in front of me.** It is fixtures and a step
index - there is no backend, no persistence, no SSE, no real tatr, and no
provider. Timestamps are literals. It exercises exactly one path per branch: one
changes-requested round rather than the every-third-round `WORK_DONE`
continuation, and no failure, cancel, restart-mid-run, or two-agents-at-once
case. It proves the information hierarchy and the legality/attribution READING
are coherent; it proves nothing about latency, streaming, or reconnect
behaviour, and it is not the implementation of any of it.

## Open questions

Not blockers; each is scoped to a v0.3.0 task rather than left implicit.

- **Retention.** The semantic log is append-only and unbounded. Retention by
  `DELETE` is proven cheap (`20260801-100405/SPIKE.md`) but the POLICY - how
  long a conversation, an activity trail, a delivery receipt is kept, and
  whether a summary pins the events it covers - is unset. It belongs with the
  WAL checkpoint tuning already deferred to 20260729-102203.
- **Summary versioning.** A rolling summary is what keeps assembled context
  bounded, but who writes it (a Scufris-side compactor, or the provider's own
  compaction read back) is undecided. Leaning Scufris-side, because a
  provider-side summary is unavailable when the session is being re-seeded -
  which is precisely when it is needed.
- **Semantic-event granularity for a streamed turn.** One event per completed
  answer is obviously right for the conversation; whether a long turn also emits
  progress events into the conversation (as opposed to activity only) changes
  the Telegram noise policy. The mockup assumes activity-only.
- **`agent_session.backend` and conversation continuity.** Making the session a
  cache means a backend switch must re-seed rather than erase. The existing
  guard makes the OLD id unreachable, which is correct; what is unspecified is
  whether the re-seed happens eagerly at switch or lazily at the next turn.
- **Multi-conversation routing.** `SCUFRIS_ORCH_SESSION_ID` routes a child's
  escalation to the spawning chat (`20260724-132713/DECISION.md`). Under
  conversation ids that becomes `SCUFRIS_CONVERSATION_ID`, which is strictly
  better (it is known before the turn, so the fresh-turn empty-parent edge
  disappears) - but the env-var rename touches the MCP wiring and belongs in its
  own task.
- **Where the guard lives.** Section 5's five checks are a service, not a route,
  and it wants the typed reader from 20260729-102158 that does not exist yet.
  Whether it is a new module or an extension of that reader's owner is the
  implementer's call.

## Next steps

**Nothing is seeded until the mockup is accepted.** `tasks/20260801-154211`
Step 4 requires it, and this task's own Step 12 says the epic's children depend
on what acceptance changes. What follows is the PROPOSED seeding, for the
acceptance conversation to confirm or redirect.

Proposed epic: **EPIC: Make Scufris an actor-aware orchestrator dashboard**,
which is `tasks/20260801-154211`'s "Epic 2: Actor-aware project coordination"
under its real name, coupled to the Project workspace rather than parallel to
it.

Proposed direction-level children (each cites this spike; none restates it):

| Direction | Reuses / depends on |
|-|-|
| Semantic conversation store: conversations, actors, ordered events, correlation and causation | new; depends on 20260729-102145 lane B |
| Channel bindings and idempotent delivery; Telegram and web as two projections | new; absorbs `telegram/approvals.py`'s in-memory `_announced` |
| Actor-aware context assembly and the provider session as a cache | new; supersedes the `format_fork_seed` path |
| Server-side flow guard over the typed tatr reader and `tatr flow -n` | **depends on 20260729-102158**, do not duplicate it |
| Project workspace: lifecycle, assignment, run, worktree, artifacts, legal next action with reasons | **re-cut 20260729-102157 and its children 102159/102200**, do not duplicate |
| Durable stage assignments binding project, task, preset, agent, run, branch, session, review, artifacts | **depends on 20260729-102205/102206**; do not re-decide the preset schema |
| Activity timeline joined to the conversation by `correlation_id` | **re-scope 20260729-102203**, do not duplicate |
| Recovery: pending gates, conversations and runs reconstructed after refresh and restart | new; the acceptance journey's step 4 |

Existing tasks to REUSE rather than re-create: 20260729-102157, 102158, 102159,
102200, 102203, 102205, 102206, and 20260729-102209 (the base plan/work/review
launch slice, to be refined not re-invented).

## Fix record

(none yet)
