# EPIC: Establish browser QA and enforce flow quality

- PRIORITY: 0
- TAGS: goal, epic, backlog, testing, flow
- KIND: EPIC
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Epic

Make "green" mean that Scufris works as a product, not only that its Python
and TypeScript units pass. Establish deterministic real-browser automation,
put it in the project's quality gates, and mechanically enforce the `$flow`
record rules that currently rely on session discipline.

## Done Means

1. A clean checkout can run deterministic Chromium tests against an isolated
   Scufris server (cmd: `cd web && npm run test:e2e`).
2. Core workflows pass at desktop and mobile widths with no page errors,
   console errors, accessibility violations, or unexpected horizontal scroll
   (test: `core-user-journeys.spec.ts`).
3. The canonical local and Nix/release gates cover Python, frontend static
   checks, frontend unit tests, production build, and browser smoke tests
   (cmd: `nix flake check`).
4. Invalid scheduling tags, closed-but-incomplete Flow State, and missing proof
   results fail repository conformance (test: `test_flow_record_conformance`).

## Child Tasks

- [ ] 20260729-102151 (p68, v0.2.0) make the mock backend stateful for
      deterministic browser QA
- [ ] 20260729-102152 (p67, v0.2.0) add a Playwright and axe browser test
      harness
- [ ] 20260729-102153 (p66, v0.2.0) automate critical desktop and mobile user
      journeys
- [ ] 20260729-102154 (p65, v0.2.0) add frontend and browser suites to
      canonical QA gates
- [ ] 20260729-102155 (p0, scufris) add repository flow lifecycle and
      scheduling conformance
- [ ] 20260729-102156 (p0, scufris) refresh project documentation and baseline
      browser polish

## Decisions

- Pending 20260729-102152: browser runner lifecycle, browser packaging, and
  accessibility integration.
- Pending 20260729-102154: which checks run on every `nix flake check` versus
  the explicit release gate.

## Manual Acceptance

- (pending) 20260729-102153: desktop and mobile journeys feel stable and
  representative of actual daily use.

## Sequencing

- Post-v0.1.0 order (2026-07-29 backlog review): FIRST of the five backlog
  epics. Highest leverage of the unscheduled work - this is a UI product whose
  unit tests say little about whether it works, and the stateful mock backend
  (20260729-102151) is what makes agent flows testable without spending
  subscription quota.
- Partially anticipated by v0.1.0: 20260729-125051 puts `nix flake check`, the
  frontend suite, and `tatr check` in CI. This epic adds the browser layer on
  top and folds it into the release gate.
- V0.2.0 prerequisite slice (2026-07-29 orchestrator readiness review):
  20260729-102151 through 20260729-102154 are pulled forward so the future
  actor-aware orchestrator lands against deterministic real-browser coverage.
  The record-conformance and documentation/polish children remain backlog, so
  this mixed-schedule epic stays OPEN and tagged `backlog`.
