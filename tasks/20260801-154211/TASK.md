# Plan and release v0.2.0: Project as the daily workspace

- PRIORITY: 109
- TAGS: release,v0.2.0,projects,flow,planning
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Story

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

## Steps

- [x] Re-cut epic 20260729-102157 as the headline v0.2.0 product epic: an
      operating surface, not an inspection surface. Reschedule its children and
      drop the ones the rewrite invalidates.
- [x] Close spike 20260729-220835 on the accepted decision and mockup.
- [x] Seed the carve epic 20260803-213242 and its five children: the workspace
      and `core`, the three host packages, and the safe half of the demolition.
- [ ] Create the `chat` package tasks: semantic events with typed actors,
      correlation and causation, monotonic `event_seq`, then idempotent
      delivery. One task per table group, each with its own example.
- [ ] Create the CONTEXT ASSEMBLY task. `tasks/20260729-220835/DECISION.md`
      section 1 makes the provider session a cache keyed
      `(conversation, backend, policy version)`, re-seeded from assembled
      context when invalid, and its Consequences warn that assembly "becomes
      code Scufris owns and must keep bounded". Nothing in this plan covered it,
      and without it the release's headline promise - the conversation survives
      `/new`, compaction, a backend switch and a restart - is unimplementable.
      Prove it with an offline example that switches backend and shows the
      conversation intact.
- [ ] Create the AGENT-REPORT-AS-QUOTATION task. This is the concrete defect the
      whole decision was written to fix: `scufris/wake.py:43` returns a machine
      prompt that `sessions/transcript.py:88-95` re-renders as `role="user"`, so
      the system speaks in the operator's voice. The code disappears with the
      orchestrator stack and no successor is named. Needs the attributed,
      untrusted-quotation rendering plus a test that an `agent:<id>` event
      cannot satisfy a stop gate.
- [ ] Create the `agents` package tasks: presets and instances, then runs and
      the provider-session binding. Include a MINIMAL activity record - tools,
      phases, exit, worktree - because it is one of the decision's four owned
      records and the acceptance journey below verifies run history from the
      Project workspace. Demoting the timeline UI to backlog is fine; demoting
      the record is not.
- [ ] Create the DURABLE RUN CLAIM task. The guard in
      `tasks/20260729-220835/DECISION.md` section 5 relies on "the existing
      `AgentRunService.launch` claim", which is
      `self._agent_runs: dict[str, str] = {}` (`orchestrator/runs.py:92`) -
      process memory. It cannot satisfy "the active run survives an application
      restart". Make it a row.
- [ ] Create the STOP-GATE CONTRACT task, before any coordinator work.
      `tatr flow -n` speaks tatr's vocabulary (`PLANNING -> WORKING`,
      `gate PLAN would run`); `PLAN_READY`, `WORK_DONE` and `LAND_READY` appear
      nowhere in `tatr`. So Scufris must map transitions to gate names AND learn
      that a run reached one - which can only come from the agent, whose reports
      are data and never instructions. Pin the mapping against the landed
      nix.dotfiles result with a red test on the exact `tatr flow -n` output it
      parses, and design how a run announces a gate without becoming its own
      approval engine.
- [ ] Create the `flow` package tasks: the typed tatr reader
      (20260729-102158), then the guard - re-read the authoritative record,
      probe with `tatr flow -n`, require an `operator` approval event, return a
      REASON on refusal - then durable assignments.
- [ ] Create the "reduce Telegram to a chat-only surface" task, scheduled BEFORE
      any deletion, and separate from the reconnection task after it. Also
      create the `packages/telegram` carve: it is in the epic's ten-unit table
      and no task builds it.
- [ ] Create the integration task: the module registry, the routers, and the
      navigation built from the registry rather than hardcoded.
- [ ] Create the Project workspace UI tasks following the mockup: lifecycle
      badges, assigned agent, active run, artifacts, and the legal next action
      with a reason on every unavailable one.
- [ ] Create the final demolition task: delete the agent/session/project/
      orchestrator stack and unlink the pages it renders, once the replacement
      is live.
- [ ] Create the Telegram re-connection task, scheduled last.
- [ ] Record the sprint order and frontier in this task once the child records
      exist.
- [ ] Drive the v0.2.0 tasks through plan, work, review, compound, and land.
- [ ] Update affected live documentation and add the notable v0.2.0 change to
      `CHANGELOG.md` in the task that changes the behavior.
- [ ] Run the canonical gates. Cut and publish v0.2.0 from `master` using
      `docs/RELEASING.md`; push `master` before the tag and verify the
      published GitHub Release.

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

## Definition of Done

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
  one bullet. Size it as real work when its tasks are cut.
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
