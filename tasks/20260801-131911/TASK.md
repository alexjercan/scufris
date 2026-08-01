# Pin the worktree-versus-main-checkout rule into the verify step

- STATUS: OPEN
- PRIORITY: 0
- TAGS: chore,process,backlog
- KIND: TASK
- FLOW STEP: BACKLOG
- PLAN STATUS: DRAFT

## Story

As a maintainer, I want the branch-versus-main-checkout trap written into the
verify step, so that an edit or a `tatr` transition meant for the branch cannot
silently land in - or act on - the main checkout.

## Notes

- Promotion of `edit-from-the-worktree-path-not-the-planning-read` (x3), reached
  2026-08-01. Ledger entry in the Pending promotions section of `LESSONS.md`.
  The promotion decision is the user's; this task is BACKLOG until they take it.
- Occurrences: 20260723-001251 and 20260726-215910 (a file Read at its
  main-checkout path then Edited in the work phase, and TASK.md planning edits
  made before `sprout new`), then 20260729-102147 in a third and nastier form -
  `tatr` acts on whichever checkout it runs in, and the `cd <worktree>` heading a
  compound Bash command persists through the rest of it, so a `tatr flow`
  appended after a `git commit` moved the WORKTREE's stale TASK.md twice under a
  task that was really in REVIEWING.
- Why the third form is worse than the first two: the branch's TASK.md is what a
  squash-merge writes over main's, so the divergence outlives the mistake and
  lands. The first two forms lose an edit; this one can regress a flow state.
- Candidate guards, cheapest first: (1) a one-line work/flow skill rule that
  every `tatr` invocation passes `-r <root>` explicitly, since cwd inside a
  compound command is not a reliable selector; (2) the same rule as a repository
  `AGENTS.md` verify-step line; (3) a wrapper or hook that refuses a bare `tatr
  flow` when cwd is a worktree whose task record differs from the main
  checkout's.
- `sprout-new-and-cd-is-denied-run-it-alone` is the adjacent lesson and should be
  cross-referenced by whatever text lands.
