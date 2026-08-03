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
3. **Build the new packages** (`20260729-102157`). `chat`, `agents`, `flow`,
   greenfield, alongside the old stack rather than on top of it. Each is proven
   ALONE by a runnable offline example plus unit tests written test-first,
   before it is wired into anything.
4. **Merge into the app.** The composition root starts serving the new packages
   through the module registry.
5. **Add the UI**, following `tasks/20260729-220835/mockup.html`.
6. **Delete the rest** - the agent/session/project/orchestrator stack and the
   pages that render it - now that its replacement is live and proven.
7. **Reconnect Telegram last.** Until then it answers the orchestrator
   conversation only and refuses agent operations with a plain message.

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
- [ ] Create the `agents` package tasks: presets and instances, then runs and
      the provider-session binding.
- [ ] Create the `flow` package tasks: the typed tatr reader
      (20260729-102158), then the guard - re-read the authoritative record,
      probe with `tatr flow -n`, require an `operator` approval event, return a
      REASON on refusal - then durable assignments.
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
