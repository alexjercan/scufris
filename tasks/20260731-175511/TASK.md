# Bump the pinned tatr to v2 and disposition the lessons ledger

- STATUS: CLOSED
- PRIORITY: 92
- TAGS: chore, v0.2.0, tooling
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

As a maintainer, I want `nix flake check` to pass while a task is IN_PROGRESS,
so that the canonical gate is usable during `/work` rather than only after a
task closes.

## Context

Commit 9d78ebe migrated every task record to the tatr v2 schema (`KIND`,
`FLOW STEP`, `PLAN STATUS`). `flake.nix` still pins tatr 0.1.0, which predates
those fields: it cannot see `PLAN STATUS: APPROVED` and reports
`unplanned-in-progress` for any IN_PROGRESS record. The gate is therefore red
for the whole duration of every task and green only between them.

Bumping the input to 0.2.0 fixes that and produced zero task-record findings,
but 0.2.0 also enforces ledger rules the current `LESSONS.md` does not satisfy:

- 10x `promotion-awaiting-decision` - a pending entry with an `(xN)` count and
  no disposition. AGENTS.md reserves promotion calls for the operator.
- 1x `bad-disposition` on `isolate-state_dir-in-tests-that-assert-config`: a
  pending entry needs an `(xN)` count.

Found while landing 20260731-171420; that task left the pin alone rather than
drag an operator-owned ledger decision into a file-size guard.

## Steps

- [x] `nix flake update tatr` (0.1.0 -> 0.2.0).
- [x] Confirm `tatr check` reports no task-record findings under the new
      version.
- [x] Put each of the 10 pending lessons to the operator: promote to the
      durable ledger, or drop it. Record the disposition on each entry.
- [x] Give `isolate-state_dir-in-tests-that-assert-config` its `(xN)` count.
- [x] Confirm the whole gate is green with a task IN_PROGRESS.

## Definition of Done

- The pinned and local tatr agree (cmd: `nix develop -c tatr version`).
- `tatr check` passes on records and ledger
  (cmd: `nix develop -c tatr check --ledger LESSONS.md`).
- The gate passes while a task is IN_PROGRESS (cmd: `nix flake check`).

## Notes

- The operator decides every promotion; do not disposition a lesson unasked.

## Outcome

Closed with no diff of its own: every Step had already been satisfied by the
tasks that ran between this record's creation and its pickup.

- `flake.nix:33` pins `github:alexjercan/tatr/v0.2.0` and `flake.lock` resolves
  `ref: v0.2.0`; the bump rode in with 1253dfd.
- All 10 pending `LESSONS.md` entries carry an operator disposition
  (PROMOTE / DEFER / ABSORBED), and the PROMOTE calls are routed to
  20260731-233221.
- `isolate-state_dir-in-tests-that-assert-config` now reads
  `(x3, DEFER 2026-07-31 ...)`, so `bad-disposition` is gone.

DoD re-verified on `master` at 95757a8, with this record itself mid-flow:
`tatr version` and `nix develop -c tatr version` both report 0.2.0,
`tatr check --ledger LESSONS.md` exits 0, and `nix flake check` reports
`all checks passed!`.
