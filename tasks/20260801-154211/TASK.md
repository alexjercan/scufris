# Plan and release v0.2.0: Project as the daily workspace

- PRIORITY: 109
- TAGS: release, v0.2.0, projects, flow, planning
- KIND: EPIC
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Epic

As the Scufris maintainer, I want v0.2.0 to be a fast vertical slice to the
target architecture - the Project workspace of
`tasks/20260729-102145/architecture.html` driven by the actor-aware
conversation model accepted in `tasks/20260729-220835/DECISION.md` - so that
the app becomes the place I operate `$flow`, and not another release of
disconnected capabilities bolted onto the current agent console.

This is a rewrite with the existing code as a starter base. Backwards
compatibility is explicitly NOT a goal. The old database is dropped, not
migrated. Legacy JSON import is deleted, not preserved. Pages that do not serve
the target flow are unlinked, not polished.

The release outcome is concrete: from one Project page, the operator can select
a task, understand authoritative lifecycle state, launch the legal next stage,
follow the assigned agent, inspect artifacts, approve human gates, and reach
land. The workspace and active work survive refresh and restart.

## Direction (2026-08-03 re-cut)

The sprint that this task previously scheduled - browser QA harness, a11y
baseline, preset spikes, run timelines, host approval UI - is now `backlog`.
None of it is on the critical path to the target app, and most of it would be
written against surfaces this rewrite deletes.

Ordering of the v0.2.0 sprint:

1. **Carve** (`20260803-213242`). Move the complete, surviving components into
   workspace packages: `core`, `host`, `hostd`, `hostctl` - in that order, since
   `hostd` imports `host.run`. Moves of tested code, so the app keeps working and
   any failure is unambiguously the carve. `hostctl` is the exception: it needs
   `EventBus`/`Supervisor` hoisted into `core` first.
2. **Delete the safe half** (`20260803-214750`). The legacy `/api/agent/*`
   router and the JSON import path have no replacement to wait for. Squash the
   migration history to one baseline.
3. **Build `chat` greenfield** (`20260729-102157`). Its tables -
   `conversation`, `event`, `delivery`, `activity` - collide with nothing, so it
   is built ALONGSIDE the old stack and proven alone by a runnable offline
   example plus unit tests written test-first.
4. **Reduce Telegram to a chat-only surface**, BEFORE anything is deleted.
5. **Replace `agents` and `flow`, one at a time, delete-then-build.** These
   CANNOT be built alongside - see the collision note below.
6. **Merge into the app.** The composition root serves the new packages through
   the module registry.
7. **Add the UI**, following `tasks/20260729-220835/mockup.html`.
8. **Delete what remains** - the orchestrator stack and the pages that render
   it.
9. **Reconnect Telegram last**, to the new conversation.

**The collision that shapes steps 3 and 5.** The carve mandates one `Base` and
one metadata. `flow` is declared owner of `projects` and `agents` owner of the
agent/run tables - and `projects` (`db/models.py:54`), `agents` (`:78`),
`agent_session` (`:109`), `agent_session_history` (`:131`) and `agent_outcome`
(`:147`) are already taken. Two classes with the same `__tablename__` on one
`DeclarativeBase` raise `InvalidRequestError` AT IMPORT: the app would not
start. `chat` is unaffected because its table names are all new.

So "greenfield alongside" holds for `chat` and only `chat`. For `agents` and
`flow` the old row classes come out in the same task that lands the new ones,
which is why step 5 is per-package rather than one deletion at the end.

**Telegram breaks at deletion, it does not degrade.**
`scufris/telegram/wiring.py:25-43` imports `agent_diagnostics`,
`agent_store.AgentStore`, `health.AgentHealth` and
`orchestrator.OrchestratorTurnService` at MODULE scope. Deleting those makes
Telegram fail to import, not answer politely - so "refuses agent operations with
a plain message" requires the step-4 reduction to happen first.

**Why demolition is split rather than done first.** The code that survives is
unambiguous and gets moved; the code that dies is exactly the code that gets
REPLACED rather than carved, so it never has to be classified at all. Deleting
it up front would buy nothing and would leave the app broken for the whole
rebuild, where a failure could be the demolition or the new logic and nobody
could tell which. Deleting it at step 6 - against a live replacement - makes
every deletion falsifiable.

What survives untouched: Stats (the host inspection surface), auth, and the
`hostd` root helper and its audit. They are not in the way.

**Epic 20260729-102157 and task 20260729-102158 leave the sprint (2026-08-04).**
Both go to `backlog`. Neither is wrong - 102157's six Done Means are the
acceptance criteria this release is judged on - but both are shaped as PAGE work
in a world the carve made PACKAGE-shaped, which is why 102157 was left with a
single in-sprint child while the package-shaped children were never minted.
102158 in particular plans edits to `scufris/projects.py`, the file that is
about to stop owning the tatr boundary. Nothing is discarded: every orphaned
Done Means is folded into a Steps bullet below, and 102158's five test names are
inherited by the TATR SDK task. This record is now the only place the v0.2.0
plan is stated.

### How work is proven

Every task in this sprint is sized to be implementable while the app around it
is half-built, and proven in this order:

1. **Unit tests, written first.** Red before green, per task.
2. **A runnable example.** `examples/<package>_*.py`, offline - temporary
   SQLite, fakes for hostd and the providers, no network. This is the PRIMARY
   proof that a package works on its own, and it is gated by
   `tests/test_examples.py`.
3. **Integration**, only once the package is wired in.

The existing examples are rewritten as packages absorb their code. Examples that
genuinely need a real NixOS box stay manual and are marked so the gate skips
them.

## Lanes (2026-08-04)

The nine-step Direction above is an ORDER. This is the same work cut into
deliverables: a lane is done when its artifact runs green in a clean checkout
and an operator can read its output without knowing the code. Every Steps bullet
below is tagged with its lane.

Deliverable kinds: **E** a runnable offline example under `examples/` with rich
output, gated by `tests/test_examples.py`; **H** an HTML explainer beside
`architecture.html`; **U** a real screen.

Lanes are SEQUENTIAL - one at a time, ordered by dependency, not preference.
Examples stay at repository root rather than under each package: `rich` is a
root dependency (`pyproject.toml:20`), and keeping demos at the root means no
package takes a dependency on a pretty-printer in order to be provable. Every
demo is offline - temporary SQLite, fake hostd, fake providers - except Lane 3,
which needs the real `tatr` binary from the dev shell.

- **Lane 0 - Carve.** In flight (`20260803-213242`). Not re-cut here.

- **Lane 1 - The conversation exists.** `chat`, built alongside the old stack.
  - **E** `chat_conversation.py` - one conversation printed as a rich
    transcript: a colour per typed actor, `event_seq`, causation as a tree.
    Mid-script it switches backend and re-prints: the transcript is identical,
    the provider session id is not.
  - **H** `chat.html` - the event model, the four owners, per-turn granularity,
    and the retention non-decision stated as a choice.

- **Lane 2 - A human can say yes.** One mechanism, two subjects.
  - **E** `operator_decision.py` - a host proposal asked on two channels;
    answered on one; the other channel's card resolves. The script then REPLAYS
    the same delivery and shows that nothing happens twice. Finally it tries to
    mint a decision from an `agent:` event and prints the refusal.
  - **H** `approval.html` - the flow, both subjects, and why the write order
    reversed.

- **Lane 3 - The repository is legible.** The tatr boundary.
  - **E** `flow_read.py` - pointed at THIS repository's `tasks/`: the real
    frontier as a rich table, then one task's artifact index, then a faked
    `tatr version` bump showing the loud refusal instead of an empty board.

- **Lane 4 - A transition is legal, or refused with a reason.**
  - **E** `flow_guard.py` - a temporary tatr repository. An illegal advance
    prints the reason. An advance without an `OperatorDecision` does not
    type-check and the script shows the refusal. With a minted one it succeeds,
    and the record is re-read and printed.

- **Lane 5 - Work runs, and survives.** `agents`. The delete-then-build lane.
  - **E** `agents_run_recovery.py` - launches a fake run, prints live activity,
    hard-exits the process, restarts, and prints the SAME claim and assignment
    restored from rows.

- **Lane 6 - It is one app.**
  - **E** `integration_registry.py` - endpoint, owning package and source record
    as a table, derived from the registry rather than from a hardcoded list.

- **Lane 7 - You can operate it.**
  - **U** the acceptance journey below, run by hand.

- **Lane 8 - Nothing legacy remains.**
  - **H** `architecture.html` updated to describe what shipped.

**Lanes 1 to 6 are not vertical slices and this record does not claim they are.**
Each ends at a terminal, not at a screen; the product is first VISIBLE in Lane 7.
The alternative - a thin UI shell in Lane 1 that grows every lane - was
considered and rejected: it buys an early demo at the price of rework in every
later lane.

## Steps

- [x] Re-cut epic 20260729-102157 as the headline v0.2.0 product epic: an
      operating surface, not an inspection surface. Reschedule its children and
      drop the ones the rewrite invalidates.
- [x] Close spike 20260729-220835 on the accepted decision and mockup.
- [x] Seed the carve epic 20260803-213242 and its five children: the workspace
      and `core`, the three host packages, and the safe half of the demolition.
- [x] (Lane 1) Create the `chat` package tasks: semantic events with typed actors,
      correlation and causation, monotonic `event_seq`, then idempotent
      delivery. One task per table group, each with its own example. Two
      questions the spike deferred are inputs here, not separate tasks:
      PER-TURN EVENT GRANULARITY (one event per turn, or one per meaningful
      thing said?) must be settled to design the `event` table at all, and the
      RETENTION NON-DECISION must be written into `chat`'s `DECISION.md` -
      v0.2.0 deletes no events, the table grows without bound, and that is a
      choice rather than an oversight.
- [ ] (Lane 2) Create the OPERATOR DECISION task, the mechanism BOTH approval kinds
      share. A flow gate and a host proposal are the same shape - something
      needs a human yes, every channel is asked, one channel answers, the other
      channel's card resolves, the waiting thing proceeds - so they get one
      mechanism with two subjects, not two approval interfaces. `chat` owns the
      asking and the answering, because it already owns actors, `event_seq` and
      idempotent delivery. `core` defines an `OperatorDecision` value type that
      only `chat` can mint, from an event whose actor is `operator`; `flow` and
      `hostctl` consume it. No new package edge and no Protocol port: `core`
      depends on nothing, so both consumers already reach it.
- [x] (Lane 1) Create the CONTEXT ASSEMBLY task. `tasks/20260729-220835/DECISION.md`
      section 1 makes the provider session a cache keyed
      `(conversation, backend, policy version)`, re-seeded from assembled
      context when invalid, and its Consequences warn that assembly "becomes
      code Scufris owns and must keep bounded". Nothing in this plan covered it,
      and without it the release's headline promise - the conversation survives
      `/new`, compaction, a backend switch and a restart - is unimplementable.
      Prove it with an offline example that switches backend and shows the
      conversation intact. Two more deferred spike questions are inputs here
      rather than tasks: SUMMARY VERSIONING and EAGER-VERSUS-LAZY RE-SEED on a
      backend switch. Assembly is what produces summaries and what decides when
      to re-seed, so both are answered by writing this task, not before it.
      Settled at understanding (2026-08-04): summarization is CUT from v0.2.0.
      The accepted decision never asks for it, `format_fork_seed` already bounds
      by windowing, and a Scufris-side summarizer calls a model - which
      contradicts proving this lane with an offline example. Summary versioning
      falls away with it; the cache key keeps `policy version` for its own
      reasons. See `tasks/20260804-115320/NOTES.md`.
- [x] (Lane 1) Create the AGENT-REPORT-AS-QUOTATION task. This is the concrete defect the
      whole decision was written to fix: `scufris/wake.py:43` returns a machine
      prompt that `sessions/transcript.py:88-95` re-renders as `role="user"`, so
      the system speaks in the operator's voice. The code disappears with the
      orchestrator stack and no successor is named. Needs the attributed,
      untrusted-quotation rendering plus a test that an `agent:<id>` event
      cannot satisfy a stop gate.
- [ ] (Lane 5) Create the `agents` package tasks: presets and instances, then runs and
      the provider-session binding. Include a MINIMAL activity record - tools,
      phases, exit, worktree - because it is one of the decision's four owned
      records and the acceptance journey below verifies run history from the
      Project workspace. Demoting the timeline UI to backlog is fine; demoting
      the record is not.
- [ ] (Lane 5) Create the DURABLE RUN CLAIM task. The guard in
      `tasks/20260729-220835/DECISION.md` section 5 relies on "the existing
      `AgentRunService.launch` claim", which is
      `self._agent_runs: dict[str, str] = {}` (`orchestrator/runs.py:92`) -
      process memory. It cannot satisfy "the active run survives an application
      restart". Make it a row. Also lands the guard's deferred "no conflicting
      active run" check and its test, which Lane 4 cannot complete.
- [ ] (Lane 4) Create the STOP-GATE CONTRACT task, before any coordinator work.
      `tatr flow -n` speaks tatr's vocabulary (`PLANNING -> WORKING`,
      `gate PLAN would run`); `PLAN_READY`, `WORK_DONE` and `LAND_READY` appear
      nowhere in `tatr`. So Scufris must map transitions to gate names AND learn
      that a run reached one - which can only come from the agent, whose reports
      are data and never instructions. Pin the mapping against the landed
      nix.dotfiles result with a red test on the exact `tatr flow -n` output it
      parses, and design how a run announces a gate without becoming its own
      approval engine.
- [ ] (Lane 3) Create the TATR SDK task. `packages/flow` gains a `tatr/` module - a
      scufris-shaped wrapper, deliberately not a generic `tatr.py`. It is the
      SOLE owner of the tatr boundary and the architecture already demands it:
      `architecture.html` lists "Structured tatr read - typed lifecycle
      metadata and safe artifact index" as one of five v0.2.0 foundations.
      Today the entire integration is `scufris/projects.py`, which scrapes
      `tatr ls` with a regex (`:52-55`) whose own comment records the incident:
      tatr added fields and "every task silently disappear[ed] from the
      Projects page instead of failing loudly". Build it as a HYBRID - parse
      `TASK.md` off disk for record fields, shell out only for what tatr
      COMPUTES (`flow -n`, `frontier`, `proofs`, `context`, `check`), since
      those already emit tab-separated records while `ls` is the human-shaped
      outlier that broke. Read surface wide, write surface narrow: no `rm`, no
      `migrate`. Assert a supported `tatr version` at startup so the next
      format change is a loud refusal at boot rather than an empty board.
      Inherit the five test names from 20260729-102158, which this replaces.
- [ ] (Lane 4) Create the FLOW GUARD task. `tasks/20260729-220835/DECISION.md` section 5
      in one place: re-read the authoritative record, probe legality with
      `tatr flow -n`, check no conflicting active run, require an operator
      approval, and on refusal return the REASON the UI renders instead of an
      unexplained disabled control. It lives at `packages/flow/guard.py` - the
      spike deferred "where the guard service lives" to v0.3.0, which is stale:
      epic 20260729-102157 required `test_flow_guard_refuses_with_reason` in
      THIS release, so the guard is v0.2.0 code and the carve already decided
      where v0.2.0 code lives. The write path is authorized by capability, not
      convention: `advance()` takes an `OperatorDecision` that only `chat` can
      mint from an operator event, so an agent cannot construct the argument.
      Back it with a boundary test that no module but the guard imports the
      write path. The "no conflicting active run" check CANNOT land here: it
      needs the durable claim, which is Lane 5. Lane 4 ships the guard without
      it, and Lane 5 adds the check plus its test. The alternative - ordering
      Lane 5 first so Lane 4 closes whole - puts the guard behind the riskiest
      lane and is rejected.
- [ ] (Lane 2) Create the HOST APPROVAL DECOUPLING task. `HostApprovalService`
      (`packages/hostctl/approvals.py`, 549 lines) owns the decision seam for
      both channels today, which makes the privileged host client the owner of
      an approval mechanism it should not have - it exists to talk to `hostd`.
      Move the decision half out to the shared OPERATOR DECISION mechanism and
      leave propose/apply/deny/audit behind. `approve()` keeps being the only
      caller of `apply`, so "an action with no approval has no route to
      execution" is preserved and STRENGTHENED: today it takes `actor: str`, a
      string anyone can fabricate, and after this it takes a minted
      `OperatorDecision`. Reverse the write order while doing it - event first,
      then apply. Today the hook fires after the row commits
      (`approvals.py:415`), so a crash between them loses the conversation
      event permanently; event-first loses only the apply, which is
      recoverable because the log says an operator approved it and `hostd`
      still holds the proposal pending. This also gives the Telegram card a
      real idempotency key `(channel, conversation_id, event_seq)` in place of
      `TelegramApprovals._announced`, an in-memory `OrderedDict` that dies on
      restart. Sequence it with the `chat` delivery task; it answers the epic's
      open "are host approvals conversation events" question with: the
      DECISION is, the proposal is not.
- [ ] (Lane 5) Create the DURABLE ASSIGNMENTS task: stage, preset, agent, run and
      worktree as a row, inserted once and restored by id, so the Project
      workspace can name the current assignment after a restart.
- [ ] (Lane 2) Create the "reduce Telegram to a chat-only surface" task,
      scheduled BEFORE any deletion, and separate from the reconnection task
      after it. Also create the `packages/telegram` carve: it is in the epic's
      ten-unit table and no task builds it. It is parked in Lane 2 for
      coherence with the approval card, but it has no real dependency on Lane 2
      and its only hard constraint is that it precede Lane 5. If Lane 2 slips,
      move it rather than letting it become Lane 5's blocker.
- [ ] (Lane 6) Create the integration task: the module registry, the routers, and the
      navigation built from the registry rather than hardcoded.
- [ ] (Lane 3) Create the ARTIFACT INDEX task: TASK, SPIKE, DECISION, REVIEW, RETRO and
      NOTES listed and opened safely inside Scufris, scoped to a registered
      project's `tasks/` directory. Reject traversal, symlink escape, unknown
      artifact names and oversized records. Inherited from 20260729-102157
      Done Means 4; note that its named proof `task-artifact-viewer.spec.ts`
      assumes Playwright, which is not in this tree - rewrite it against
      vitest.
- [ ] (Lane 7) Create the three Project workspace UI tasks, sliced by CAPABILITY rather
      than by screen so each one is a full promise from store to pixel:
      **(1) PROJECTION** - the board and the task detail render tatr truth
      read-only: lifecycle badges, priority, flow state, dependencies,
      artifacts. **(2) ACTIONS** - the legal next action, a reason on every
      unavailable control, and the operator stop-gate cards that approve a
      transition. **(3) RECOVERY** - the workspace, the pending stop gate and
      the active run survive a refresh and an application restart, with the
      same blocking question reconstructed from the log rather than from
      memory. They map onto 20260729-102157 Done Means 1, 3 and 5. Task 1
      alone is the inspection surface that epic explicitly rejects, so it is
      not a stopping point: 1 and 2 land in the same release or neither does.
- [ ] (Lane 8) Create the final demolition task: delete the agent/session/project/
      orchestrator stack and unlink the pages it renders, once the replacement
      is live.
- [ ] (Lane 8) Create the Telegram re-connection task, scheduled last.
- [ ] (Lane 8) Create the CLEANUP SWEEP task, after the final demolition. This release
      deletes roughly a third of the application, so stale prose is a certainty
      rather than a risk: `README.md` sections describing removed surfaces,
      comments naming deleted modules, `docs/`, and the module docstrings that
      still describe the orchestrator. Big enough to be a task, not a line in
      somebody else's commit.
- [ ] Record the sprint order and frontier in this task once the child records
      exist.
- [ ] Drive the v0.2.0 tasks through plan, work, review, compound, and land.
- [ ] Update affected live documentation and add the notable v0.2.0 change to
      `CHANGELOG.md` in the task that changes the behavior.
- [ ] Run the canonical gates. Cut and publish v0.2.0 from `master` using
      `docs/RELEASING.md`; push `master` before the tag and verify the
      published GitHub Release.

## Child Tasks

Minted records, by lane. A lane's deliverable task is listed last and the lane
is not done until it is. Unminted lanes are the unchecked bullets in Steps
above; their spec paragraphs move into the child records as they are cut.

Lane 0 - Carve: `20260803-213242`, CLOSED.

Lane 1 - The conversation exists:

- [ ] 20260804-115256 (p100) record the chat conversation and event tables with
      typed actors. Settles per-turn granularity and writes the retention
      non-decision. Ships `examples/chat_conversation.py` minimal, because the
      member gate goes red the moment `packages/chat` exists.
- [ ] 20260804-115319 (p99) deliver chat events to every channel exactly once.
      Blocked by 115256. Lane 2's host approval decoupling depends on this
      table.
- [ ] 20260804-115320 (p98) assemble provider context from the semantic
      conversation. Blocked by 115256. Bounds context with a window;
      summarization is cut from the release and recorded as a deferral.
- [ ] 20260804-115321 (p97) render agent reports as attributed quotations.
      Blocked by 115256 and 115320.
- [ ] 20260804-115322 (p96) DELIVERABLE - prove Lane 1 with the conversation
      demo and `chat.html`. Blocked by all four.

## Acceptance journey

One end-to-end journey, from the mockup's steps:

1. Start with no task, then create or select one.
2. Run planning to PLAN_READY. Exercise "Stop and let me decide," prove no
   transition, resume cold, then approve the move to PLANNED.
3. Launch the work agent and observe attributable WIP. At initial WORK_DONE,
   approve the move from WORKING to REVIEWING.
4. Refresh and restart Scufris without losing conversation, assignment, run
   state, or the ability to reconstruct a pending stop gate.
5. Run review and exercise changes-requested fixes returning directly to
   review.
6. Approve review and prove it proceeds directly to compound without another
   stop. Compound closes the task and returns LAND_READY.
7. Approve landing, then verify final tatr truth, run history, semantic
   conversation, proofs, and artifacts from the Project workspace.

## Scope Exclusions

- Backwards compatibility of any kind: no data migration, no legacy import, no
  deprecation window, no compatibility routes.
- Everything moved to `backlog` in the 2026-08-03 re-cut: browser QA harness,
  a11y baseline, preset spikes, run activity timeline, host approval UI.
- Capability/plugin epic 20260729-102204 and its children.
- Rich generic artifact framework 20260729-102210.
- Research swarm epic 20260729-102218.
- A repository editor, terminal emulator, or tmux replacement.
- A second workflow store, run log, approval engine, or conversation history.

## Done Means

- The accepted sprint plan lists every v0.2.0 epic and task with priorities,
  release tags, dependencies, scope guards, and falsifiable proofs
  (manual: user approves this task's recorded sprint frontier).
- Epic 20260729-102157 claims an operating Project workspace and all scheduled
  v0.2.0 children are at `FLOW STEP: DONE`
  (cmd: `tatr ls --sort priority > /tmp/scufris-v020-tasks && ! rg -P 'FLOW STEP: (?!DONE)[A-Z]+, TAGS: [^]]*v0\.2\.0' /tmp/scufris-v020-tasks`).
- No legacy surface survives the cut: no `/api/agent/*` routes and no JSON
  import path
  (cmd: `! rg -q 'legacy_agent|db\.legacy|legacy_import' --glob '!tasks/**' --glob '!CHANGELOG.md' .`).
- Every package is proven alone by a gated offline example before it is wired in
  (cmd: `python -m pytest tests/test_examples.py`).
- The acceptance journey above runs end to end against the real app
  (manual: user runs the v0.2.0 acceptance journey).
- Canonical checks and release packages pass
  (cmd: `nix flake check && nix build .#scufris .#scufris-web && cd web && npm run ci`).
- Version, changelog, and release metadata agree before tagging
  (cmd: `scripts/check-release-ready.sh v0.2.0`).
- The v0.2.0 release workflow passes and the public release is inspectable
  (cmd: `gh release view v0.2.0`).

## Notes

- Architecture source: `tasks/20260729-102145/architecture.html`, especially
  "Project as the daily workspace" and "Run the flow SDLC on one task".
- Conversation and flow-authority source:
  `tasks/20260729-220835/DECISION.md`; UI and interaction source:
  `tasks/20260729-220835/mockup.html`.
- Flow contract: nix.dotfiles task 20260801-155024 defines the four
  context-cut stop gates - PLAN_READY, initial WORK_DONE, every-third
  review-continuation WORK_DONE, LAND_READY - and the direct review routes.
  Re-read its landed result before writing the coordinator tasks.
- Six questions `tasks/20260729-220835/DECISION.md` deferred "to v0.3.0 tasks"
  now have no release to land in, because the re-cut collapsed v0.3.0 into
  v0.2.0: retention policy, summary versioning, per-turn event granularity,
  eager-versus-lazy re-seed on a backend switch, the `SCUFRIS_ORCH_SESSION_ID`
  rename, and where the guard service lives. Retention and re-seed are
  load-bearing for `chat` and fold into its task list; the rest need a v0.3.0
  container rather than being addressed by neither release.
- `tasks/20260729-102157` Done Means 4 cites `task-artifact-viewer.spec.ts`, but
  `web/package.json:15` runs vitest and the tree has no Playwright dependency
  and no `*.spec.ts`. Rewrite that proof against vitest, or restore a minimal
  browser harness task from backlog. As written it cannot be satisfied.
- The UI is ~14.3k lines across 73 files in `web/src`, and step 7 is currently
  one bullet. Size it as real work when its tasks are cut. Lane 7 is likely
  larger than Lanes 1 to 6 combined; "three UI tasks" is a capability slicing,
  not an estimate, and splitting the backend into tidy lanes does not touch this
  risk. Expect it to split further at planning time.
- Lane risks, recorded at the 2026-08-04 lane cut:
  - **Lane 5 is the long red period.** The table collision means the old agent
    rows come out in the same task the new ones land. The app does not start in
    between and no demo can run. It is also the only lane where a failure is
    ambiguous, because deletion and new logic land together. Every other lane
    can be abandoned half-done; this one cannot.
  - **Lane 2 re-opens work the carve just called complete.** `HostApprovalService`
    is 549 passing lines. The decoupling is correct and strengthens the security
    property, but it is a refactor of finished code and can quietly double.
  - **Pretty output is not a proof.** `tests/test_examples.py` judges each
    example by its EXIT CODE, so rich tables nobody asserts on rot into
    decoration that still exits 0. Every lane demo needs at least one assertion
    that fails loudly when the claim breaks; the rendering is for the operator,
    not for the gate.
  - **HTML explainers are the easiest place to burn a week.** Three are
    scheduled - the event model, the approval flow, and the final architecture
    update - deliberately not one per lane.
- `tests/test_examples.py` already requires every workspace member to name an
  offline example that imports it (`EXAMPLES_BY_MEMBER`). "Each lane ends in a
  runnable proof" is therefore the existing convention, not a new one: a new
  package under `packages/` cannot appear without an entry there.
- `tasks/20260729-220835/DECISION.md` section 2's central invariant - "no
  surface reads two of these for the same fact" - has no proof anywhere. The
  import rule is orthogonal: a router can legally read two packages' public APIs
  for one fact. The integration task should declare each endpoint's source
  record and assert it in the route-contract test.
- Repository task files and tatr remain authoritative for lifecycle truth.
  Scufris projects that truth and enforces fresh server-side launch guards.
- The Project page is a control surface over repository work. It does not
  replace the repository, worktree tooling, provider-native transcript, or
  privileged host audit.
- Release procedure: `docs/RELEASING.md`. Release only from the main checkout,
  on `master`, inside `nix develop`.
- History: this task was scheduled as the v0.3.0 release plan until
  2026-08-03. It was retitled and re-cut into v0.2.0 when the maintainer chose
  to cut the intervening polish release and go straight at the target
  architecture.
