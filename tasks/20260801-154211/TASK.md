# Plan and release v0.3.0: Project as the daily workspace

- STATUS: OPEN
- PRIORITY: 109
- TAGS: release,v0.3.0,projects,flow,planning
- KIND: TASK
- FLOW STEP: BACKLOG
- PLAN STATUS: DRAFT

## Story

As the Scufris maintainer, I want one accepted, dependency-ordered v0.3.0
sprint and release plan centered on "Project as the daily workspace", so that
the release turns Projects into the normal place to operate `$flow` instead of
shipping another disconnected set of backend and UI capabilities.

The release outcome is concrete: from one Project page, the operator can
select a task, understand authoritative lifecycle state, launch the legal next
stage, follow the assigned agent, inspect artifacts, approve human gates, and
reach land. The workspace and active work survive refresh and restart.

## Steps

- [ ] Re-read `tasks/20260729-102145/architecture.html`, the current v0.2.0
      release frontier, epic 20260729-102157, spike 20260729-220835, and every
      candidate task before changing schedules or dependencies.
- [ ] Reconcile the Project coordinator and acceptance journey with the final,
      landed explicit stop-gate contract from nix.dotfiles task
      20260801-155024. Preserve PLAN_READY, initial WORK_DONE, every-third
      review-continuation WORK_DONE, and LAND_READY as blocking user decisions;
      preserve direct review-fix loops and APPROVE -> COMPOUNDING behavior.
- [ ] Audit the v0.2.0 entry criteria below. Keep completed foundation work in
      v0.2.0; explicitly carry an unfinished prerequisite into v0.3.0 only when
      the v0.2.0 release cut requires it.
- [ ] Complete and obtain user acceptance for the actor-aware conversation and
      flow-control decision and mockup in 20260729-220835 before creating its
      implementation epic or production schema tasks.
- [ ] Re-cut epic 20260729-102157 as the headline v0.3.0 product epic. Update
      its goal and Done Means from an inspection surface to an operating
      surface, then schedule its retained and new children in dependency order.
- [ ] Seed the narrow actor-aware project-coordination epic selected by
      20260729-220835. Keep it coupled to the Project workspace outcome rather
      than creating a parallel product conversation, activity log, or workflow
      truth store.
- [ ] Refine 20260729-102209 and adopt only the base plan/work/review launch
      slice needed by v0.3.0. Leave the general specialist proposal editor,
      plugin capabilities, and general capability approvals in backlog.
- [ ] Create the missing tasks listed below with falsifiable Done Means,
      explicit dependencies, one release tag, relative priorities, affected
      live-document updates, and browser/API integration proofs.
- [ ] Record the sprint order and frontier in this task after the child records
      exist. Resolve dependency cycles and scope overlap before implementation
      begins.
- [ ] Drive the v0.3.0 tasks through plan, work, review, compound, and land.
      Re-plan from accepted decisions and test evidence, not from stale dates.
- [ ] Add one release acceptance journey that exercises the complete Project
      lifecycle at desktop and mobile widths with the deterministic mock
      backend, including refresh and application restart recovery.
- [ ] Update affected live documentation and add the notable v0.3.0 change to
      `CHANGELOG.md` in the task that changes the behavior.
- [ ] Run the canonical backend, frontend, browser, and package gates. Cut and
      publish v0.3.0 from `master` using `docs/RELEASING.md`; push `master`
      before the tag and verify the published GitHub Release.

## Proposed v0.3.0 Cut

### Entry criteria from v0.2.0

These are foundations, not the v0.3.0 product claim. Finish them before the
workspace implementation when possible:

- 20260729-102145: durable and backend-truthful state and services.
- 20260729-102158: structured lifecycle and artifact task metadata.
- 20260729-102203: durable agent-run activity and hierarchy.
- 20260729-102205 and 20260729-102206: reusable stage-agent presets.
- 20260729-102151 through 20260729-102154: deterministic browser QA and gates.
- 20260729-102202: responsive and accessibility baseline.
- 20260729-220835: accepted actor-aware conversation and flow-control model.

### Epic 1: Project as the daily workspace

Promote 20260729-102157 to the headline v0.3.0 epic.

Retain and schedule:

- 20260729-102159: filterable flow task board.
- 20260729-102200: in-app task artifact viewer.
- 20260729-102203: run activity timeline, if not completed in v0.2.0.

Add tasks for:

- Project task-detail workspace joining lifecycle, dependencies, assignment,
  worktree, active run, artifacts, and legal next action.
- Server-side flow guards that re-read authoritative tatr records before every
  transition or launch.
- Durable task/stage assignments linking the project, task, preset, agent,
  run, worktree, branch, provider session, proofs, review, and artifacts.
- Project lifecycle controls for explicit PLAN_READY, initial WORK_DONE,
  every-third review-continuation WORK_DONE, and LAND_READY stop gates. A stop
  choice changes no state. Ordinary review fixes return directly to review;
  APPROVE proceeds directly to compound; compound closes the task before the
  landing gate.
- Clear reasons for unavailable actions instead of unexplained disabled
  controls.
- Restart-safe workspace state and duplicate-free live SSE recovery.

### Epic 2: Actor-aware project coordination

Create this epic only after 20260729-220835 is accepted.

Add tasks for:

- Scufris-owned semantic conversation with actor attribution.
- Project conversation projection shared by browser and Telegram.
- Separate semantic conversation, technical activity, provider transcript,
  and enforcement audit records with stable correlation.
- Project flow coordinator for plan, work, review, compound, and land.
- Resolution and launch of stage-specific agents from accepted project preset
  policy.
- Provider-session, conversation, task, assignment, and run recovery after
  refresh or restart.
- The base plan/work/review slice refined from 20260729-102209.

### Release acceptance task

Add one end-to-end task covering this exact scenario:

1. Start with no task, then create or select one.
2. Run planning to PLAN_READY. Exercise "Stop and let me decide," prove no
   transition, resume cold, then approve the move to PLANNED.
3. Launch the work agent and observe attributable WIP. At initial WORK_DONE,
   approve the move from WORKING to REVIEWING.
4. Refresh and restart Scufris without losing conversation, assignment, run
   state, or the ability to reconstruct a pending stop gate.
5. Run review and exercise changes-requested fixes returning directly to
   review. Exercise the every-third-round WORK_DONE continuation gate.
6. Approve review and prove it proceeds directly to compound without another
   stop. Compound closes the task and returns LAND_READY.
7. Approve landing, then verify final tatr truth, run history, semantic
   conversation, proofs, artifacts, and lessons from the Project workspace.

## Scope Exclusions

- Full capability/plugin epic 20260729-102204.
- Plugin manifests, secret references, and general capability grants.
- Rich generic artifact framework 20260729-102210.
- Research swarm epic 20260729-102218.
- Email, calendar, PDF, PPTX, and other personal automation integrations.
- A repository editor, terminal emulator, or tmux replacement.
- A second workflow store, run log, approval engine, or conversation history.

## Definition of Done

- The accepted sprint plan lists every v0.3.0 epic and task with priorities,
  release tags, dependencies, scope guards, and falsifiable proofs
  (manual: user approves this task's recorded sprint frontier).
- Epic 20260729-102157 claims an operating Project workspace and all scheduled
  v0.3.0 children are at `FLOW STEP: DONE`
  (cmd: `tatr ls --sort priority > /tmp/scufris-v030-tasks && ! rg -P 'FLOW STEP: (?!DONE)[A-Z]+, TAGS: [^]]*v0\.3\.0' /tmp/scufris-v030-tasks`).
- One browser journey proves all explicit stop gates and their no-transition
  choice, direct review-fix and APPROVE -> COMPOUNDING routes, land approval,
  refresh, restart, mobile layout, and correct actor/run/task attribution
  (test: `project-flow-release.spec.ts`).
- The Scufris repository is practical to operate through the Project page
  without opening task records in an external editor for routine flow work
  (manual: user runs the v0.3.0 acceptance journey).
- Canonical checks and release packages pass
  (cmd: `nix flake check && nix build .#scufris .#scufris-web && cd web && npm run ci`).
- Version, changelog, and release metadata agree before tagging
  (cmd: `scripts/check-release-ready.sh v0.3.0`).
- The v0.3.0 release workflow passes and the public release is inspectable
  (cmd: `gh release view v0.3.0`).

## Notes

- Architecture source: `tasks/20260729-102145/architecture.html`, especially
  "Project as the daily workspace", "Run the flow SDLC on one task", and
  "What builds on what".
- Flow contract update (2026-08-01): nix.dotfiles task 20260801-155024 on
  `feat/flow-explicit-gates` defines the four context-cut stop gates and the
  direct review routes. At inspection time it was REVIEWING with a
  REQUEST_CHANGES verdict. Its open blockers cover committing an approved
  transition before the context cut and resolving the active worktree before
  authoritative resume reads. Re-read the reviewed, landed result before
  creating v0.3.0 implementation tasks; do not freeze the inspected draft.
- Repository task files and tatr remain authoritative for lifecycle truth.
  Scufris projects that truth and enforces fresh server-side launch guards.
- The Project page is a control surface over repository work. It does not
  replace the repository, worktree tooling, provider-native transcript, or
  privileged host audit.
- Board plus artifact viewer alone is insufficient. v0.3.0 must let the
  operator advance real work through the complete flow lifecycle.
- Release procedure: `docs/RELEASING.md`. Release only from the main checkout,
  on `master`, inside `nix develop`.
