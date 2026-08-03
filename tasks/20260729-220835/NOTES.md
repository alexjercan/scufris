# Notes: define the actor-aware orchestrator conversation and flow-control model

Understanding brief for the spike. Findings and grounding only; the spike's own
comparison, decisions and mockup are still its work.

## What changes

Before: an operator drives the flow by hand. Each phase is a fresh agent
context, re-typed. Lifecycle truth lives in `tasks/` and git; the dashboard
reads a best-effort `tatr ls` listing (`scufris/projects.py:264`) and shows
titles. The conversation with an agent IS its provider session
(`AgentSessionRow.current_session_id`), reachable only from the Agents console.
Approvals exist for host actions, not for flow gates. Nothing links a task to
the run that advanced it, the session that did the work, or the commits it
produced.

After: a Project page is the operating surface. One durable Scufris
conversation carries the human-readable narrative across many disposable
provider sessions. The operator sees who said what, what is running, which
decision is pending, which action is legal next, and can drill to the native
transcript or the technical activity. The same conversation reaches Telegram.
Gates are answered by a human, or by a recorded auto-approve policy when the
run is started unattended.

### Findings that the spike should treat as established

1. **Durable state is a sufficient handoff.** A fresh context reconstructs a
   whole phase from `tasks/`, git and the worktree. The mediator can therefore
   be logically stateless, and sessions disposable, without losing anything.
   This is the strongest argument against storing the full provider transcript
   as product truth: nothing downstream needs it to continue.
2. **Lifecycle gates are mechanically probeable.** `tatr flow -n` and
   `sprout sync/land -n` run the real preconditions and refuse without
   changing anything. Server-side launch guards need no agent turn to decide
   whether the next stage is legal, and a refusal carries usable text.
3. **An agent's self-report is a hint, not authority.** Any status an agent
   emits must be cross-checked against ACTIVITY, GATES, RESOLUTION and git
   before it is acted on; disagreement is a stop, not a tiebreak. This is the
   concrete form of "distinguish untrusted agent reports from operator
   instructions".
4. **Progress is falsifiable.** A content hash over ACTIVITY + GATES +
   RESOLUTION + branch head + worktree diff + review verdict answers "did this
   turn change anything durable" without trusting prose. Same primitive
   generalizes to event idempotency keys.
5. **An approval must always be a record.** Auto-approve is a policy that
   answers the record, never a code path that skips it. Otherwise an unattended
   run and a supervised run produce different histories, and the unattended one
   is unreplayable and unauditable. Policy is per-gate, not global.
6. **Worktrees isolate work; the main checkout is a single-writer resource.**
   Two tasks in two sprout worktrees have disjoint `tasks/<id>` records and
   disjoint files - real isolation. Landing and task creation touch the main
   checkout and must be serialized.
7. **Messages address the conversation, not the session.** A live session is
   frequently absent - between phases, after a rotation, after a restart. A
   user message is persisted to the conversation first, then delivered to a
   live session if there is one, otherwise folded into the next session's
   context assembly. The provider session becomes a cache to replay into, not
   a place things are stored.
8. **Live injection is a capability the backends do not have.**
   `AgentBackend.stream` (`scufris/backends/base.py:99`) is one-shot: prompt in,
   events out, resuming `session_id`. Nothing can push an unsolicited message
   into a turn in flight - `scufris/wake.py` says so explicitly and works
   around it by deferring and batching. Mid-turn injection needs a streaming
   input channel per backend, and is a prerequisite, not an assumption.

## Surfaces

Read and modelled by the spike; not all are edited by it.

| Path | Why |
|-|-|
| `scufris/db/models.py:92` | `AgentSessionRow` already binds agent -> backend -> current session and records `parent_agent_id`/`parent_session_id`. The conversation/actor records extend this shape rather than replace it. |
| `scufris/db/models.py:118` | `AgentSessionHistoryRow` - ordered session history per agent. Precedent for "session ids are rows, not a JSON list". |
| `scufris/db/models.py:138` | `AgentOutcomeRow` - durable terminal outcome, observable by a process that was not watching. The activity stream generalizes this. |
| `scufris/orchestrator/runs.py:1` | `AgentRunService`: launch, one-run-per-agent guard, serialize key, cancel, event relay, completion fan-out. The only place "busy" is defined; flow launch guards belong beside it. |
| `scufris/wake.py:1` | Deferral + batching of sub-agent questions while the orchestrator is mid-turn. Already the "hold it and fold it into the next turn" mechanism finding 7 needs. |
| `scufris/host_approvals.py:1` | The decision seam: rules once, each surface only says WHO decided. Template for flow-gate approvals across web and Telegram. |
| `scufris/telegram/approvals.py`, `scufris/telegram/turn.py` | Existing dual-surface decision path and turn plumbing. |
| `scufris/sessions/transcript.py`, `scufris/sessions/rollout.py` | Provider transcript readers. Evidence that the native transcript is readable on demand, so it need not be copied. |
| `scufris/sessions/steering.py` | Prompt-borne steering; where context assembly's system/policy layer lands. |
| `scufris/projects.py:264` | `read_project_tasks` shells out to `tatr ls`, best-effort, never raises. Fine for a listing, insufficient for flow control - guards need authoritative, error-propagating reads. |
| `scufris/backends/base.py:99` | The one-shot `stream` contract, finding 8. |
| `scufris/eventbus.py`, `scufris/api/sse.py` | Existing live event delivery to the web surface. |
| `web/src/project-detail-view.ts` | Today's Projects drill-in: a task list. The workspace the spike mocks replaces this. |
| `tasks/20260729-220835/mockup.html` | New. Fixture-driven static artifact, no production wiring. |
| `tasks/20260729-220835/SPIKE.md`, `DECISION.md` | New. The comparison and the accepted load-bearing choices. |

## Data and interfaces

Logical records the spike must decide. Signatures are indicative shapes for the
decision, not a schema to implement here.

`Workspace` is the unit of work - one tatr task plus the sprout worktree it
acquires. It owns exactly one `Conversation` and many `AgentRun`s. A
`Conversation` is the whole semantic narrative in one thread: operator,
orchestrator and every specialist, separated by `Actor` rather than by thread.
A run's provider session is disposable and belongs to the run, so many sessions
per workspace is expected and is NOT many conversations.

```
Project(id, cwd, name, ...)                              -- exists today

Workspace(id, project_id, task_id, worktree?, branch?, created_at, closed_at?)
  -- one per tatr task; worktree/branch set at sprout, cleared at land

Conversation(id, project_id, workspace_id?, parent_conversation_id?, title,
             created_at)
  -- workspace_id NULL is the project-level conversation ("/")
  -- parent_conversation_id: "/" is where a workspace conversation is spawned

Actor(id, kind: operator|orchestrator|specialist|system, agent_id?, label)

ConversationEvent(id, conversation_id, seq, actor_id, kind, body,
                  run_id?, correlation_key, idempotency_key, created_at)

ChannelBinding(conversation_id, channel: web|telegram, target, enabled)
Delivery(event_id, channel, external_id, delivered_at)   -- dedup

AgentRun(id, workspace_id, conversation_id, agent_id, stage,
         phase_before, phase_after, started_at, ended_at, outcome)
ProviderSessionBinding(run_id, backend, session_id, seq)
  -- seq: a run that rotates context spans several sessions
ActivityEvent(run_id, seq, kind, payload)                -- technical stream

Approval(id, workspace_id, conversation_id, gate, requested_at, decided_at,
         decided_by, decision, policy: human|auto, refusal_text?)
WorkflowAssignment(workspace_id, stage, agent_id, preset_id?)
Artifact(workspace_id, kind, path, commit?, branch?)
```

### Relationships

`||` mandatory one, `o|` optional one, `<` many.

```
  Project ||----< Workspace ||----|| Conversation ||----< ConversationEvent
     |                |                   |  |                    |
     |                |                   |  |                    o
     |                |                   |  +----< ChannelBinding|
     |                |                   |             |         |
     |                |                   |             +--< Delivery
     |                |                   |
     +----< Conversation ("/" , workspace o| NULL)
                       ^
                       |  parent_conversation_id (self, optional)
                       +-- a workspace conversation is spawned from "/"

  Workspace ||----< AgentRun ||----< ProviderSessionBinding
      |                 |                    |
      |                 |                    +-- backend + native session id;
      |                 |                        the transcript stays native
      |                 ||----< ActivityEvent
      |                 |
      |                 o|---- ConversationEvent.run_id
      |                        (which run produced this semantic event)
      |
      ||----< Approval
      ||----< WorkflowAssignment
      ||----< Artifact

  Actor ||----< ConversationEvent
  Actor o|---- AgentRun.agent_id      (system/operator actors have no run)
```

Reading the cardinalities that matter:

| Relation | Cardinality | Why |
|-|-|-|
| Workspace -> Conversation | 1 : 1 | the narrative of one task is one thread |
| Conversation -> Workspace | 1 : 0..1 | `/` has no task yet; this asymmetry is why they are two tables |
| Workspace -> AgentRun | 1 : N | every stage, every review-fix loop, every retry |
| AgentRun -> ProviderSession | 1 : N | a run that rotates on context spans several |
| ConversationEvent -> AgentRun | N : 0..1 | operator messages belong to no run |

Interfaces the spike must define, in the shape the code already uses:

```python
# authoritative, error-propagating - unlike read_project_tasks
def read_task_state(root: Path, task_id: str) -> TaskState: ...
    # ACTIVITY, GATES, RESOLUTION, worktree, branch, review verdict

def probe_transition(root: Path, task_id: str) -> ProbeResult: ...
    # dry run; ok | refused(text). Changes nothing.

def legal_actions(state: TaskState) -> list[FlowAction]: ...
    # what the Project page may offer, and why the rest are greyed out

def post_message(conversation_id: str, actor: Actor, body: str) -> ConversationEvent: ...
    # persists FIRST; delivery is a consequence, not a precondition

def assemble_context(conversation_id: str, task_id: str) -> Prompt: ...
    # policy + versioned summary + recent semantic events + pending decisions
    # + queued operator messages + available capabilities

def durable_fingerprint(root: Path, task_id: str) -> str: ...
```

## Sketches

Illustrative only.

The approval seam, mirroring `host_approvals`:

```
+    approval = approvals.request(conversation_id, task_id, gate="PLAN_READY",
+                                 body=probe.refusal_text or summary)
+    if policy.auto(gate):
+        approvals.decide(approval.id, actor=SYSTEM_POLICY, decision=APPROVE)
+    # either way the record exists, and the event stream is identical
```

Delivery, not storage:

```
-    backend.stream(settings, prompt, session_id=sid)
+    event = post_message(conversation_id, operator, text)   # durable, ordered
+    if live := runs.active(conversation_id):
+        live.inject(event)          # requires the streaming input channel
+    else:
+        pass                        # assemble_context picks it up next session
```

Guard before launch:

```
+    state = read_task_state(root, task_id)
+    if action not in legal_actions(state):
+        raise IllegalTransition(state, action)
+    probe = probe_transition(root, task_id)     # tatr/sprout own legality
```

## Shape

```
  operator (web)        operator (telegram)
        |                       |
        +----------+------------+
                   v
        +---------------------------+
        |  Scufris conversation     |  durable, ordered, replayable
        |  (semantic events)        |  <- the product's source of truth
        +---------------------------+
           |          |          |
           |          |          +--> Delivery (per channel, deduped)
           |          |
           |          +--> Approval records (gate pending/decided, human|auto)
           |
           v
        +---------------------------+
        |  mediator / context       |  logically stateless
        |  assembly                 |  reads durable state every time
        +---------------------------+
                   |
                   v  launch (guarded)
        +---------------------------+        +----------------------+
        |  AgentRun                 |------->| provider session     |
        |  activity events          |  binds | (native transcript)  |
        +---------------------------+  by id | cache, not truth     |
                   |                          +----------------------+
                   v  works in
        +---------------------------+
        |  sprout worktree          |  isolated per task
        |  tasks/<id> + branch      |
        +---------------------------+
                   |
                   v  probed read-only, never guessed
        +---------------------------+
        |  tatr + git = workflow    |  authority for ACTIVITY/GATES/RESOLUTION
        |  truth, enforcement audit |
        +---------------------------+
                   ^
                   |  serialized: land + task creation touch the main checkout
```

Four distinct streams, one origin each:

| Stream | Origin | Audience |
|-|-|-|
| semantic conversation | mediator + operator | the human, both channels |
| technical activity | run events, tokens, rotations | drill-in, debugging |
| enforcement audit | probe refusals, guard rejections, approvals | audit, not chat |
| provider transcript | the backend's own rollout | on-demand drill-in only |

## Consequences and open questions

Costs:

- A second conversation model beside `AgentSessionRow`. Two ways to reach an
  agent - the Agents console (native session) and the Project conversation -
  and the spike must say which one owns a given interaction.
- Mid-turn injection needs a new backend capability. Until it exists, operator
  messages land between sessions only. Mock backend must model both.
- Approval records add a parked state to the run lifecycle. `AgentRunService`
  currently knows running/idle; it will need pending-decision.
- Serialization around the main checkout is a new global constraint; it caps
  concurrent landings at one regardless of worktree count.

Forecloses:

- Storing the provider transcript as product truth. Once the semantic
  conversation is authoritative, a backend swap must not lose history, and
  copying a backend-shaped transcript would reintroduce the coupling.
- Auto-approve as a bypass. Recorded-policy-answers-record is the only shape
  that keeps unattended runs replayable.

Decided, not open:

- **Operator messages queue; they never interrupt a turn.** Persist to the
  conversation, show as pending in both surfaces, fold into the next session's
  context assembly. Interrupt was rejected as more useful but much harder - the
  agent is mid-tool or mid-commit - and finding 8 makes it a backend capability
  that does not exist. The mockup must show the pending-message affordance.
- **`Workspace` is the unit of work, and it owns exactly one conversation.**
  A workspace is one tatr task plus the sprout worktree it acquires. The
  worktree is an attribute, not an identity: a task has no worktree during
  understanding and planning, a rewind or re-sprout gives the same task a new
  one, and the record must outlive the worktree that landing removes.
- **One conversation per workspace, not one per session.** The conversation is
  the whole semantic narrative - operator, orchestrator and every specialist -
  separated by `Actor`, not by thread. Many provider sessions per workspace is
  expected: a session belongs to an `AgentRun`, is disposable, and is a cache.
  Splitting the conversation per session would make the narrative unreadable
  and destroy replay.
- **One project-level conversation as well.** It is what `/` is, and it has no
  workspace. A goal is discussed there before any task id exists, and the
  workspace conversation is spawned from it once the id is minted - which the
  "no task -> planning agent" scenario requires.

Assumptions recorded rather than blocking:

- Auto-approve is per-gate policy on a run, defaulting to all-human.

Open questions for the spike:

- Which stage agents are reusable presets vs ephemeral, and which are hidden
  from the conversation entirely.
- Whether the enforcement audit is a projection of the conversation or its own
  table. Retention differs, which argues for its own.
- Correlation key shape: workspace -> run -> session -> commit is a chain, but
  a review-fix loop revisits the same workspace with several runs at the same
  stage, so `stage` alone does not identify a run.
- Restart recovery: what a run in flight becomes when the process dies, and
  whether a live provider session is re-attachable or always abandoned.
- Compaction policy for the versioned summary, and who writes it.
