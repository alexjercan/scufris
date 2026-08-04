# Promote the recurring lessons into repository guards

- PRIORITY: 0
- TAGS: process,backlog,lessons,docs
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Story

As a maintainer, I want the lessons that have now recurred three or more times
written into the repository's own guard surfaces, so that the next task is
stopped by a rule rather than by paying the same cost again.

## Steps

- [ ] `run-the-check-against-the-pre-move-file-before-recording-a-cause` ->
      `AGENTS.md` verify clause: when a gate fires on MOVED code, measure it -
      `git show <base>:<path>` into a scratch file (never a checkout from a
      worktree) and run the check against that before recording a cause.
- [ ] `format-only-the-files-you-edited-not-whole-dirs` -> `AGENTS.md` verify
      clause: scope every `ruff format` / `ruff check --fix` / `prettier --write`
      to the files you edited, never `.` or a whole directory.
- [ ] `a-patch-target-string-can-survive-a-split-and-stop-patching` -> a
      pre-split survey clause, or a script that greps `setattr("<module>.` for
      every module a split touches, since the failure is silent when the facade
      binds the name.
- [ ] `nix-devshell-import-resolves-to-cwd-source` -> `AGENTS.md` verify clause:
      verify branch code with `python -m pytest` and `cd <tree> && python -m
      scufris`, never the bare console script from elsewhere.
- [ ] `protocol-signature-change-hits-the-doubles` -> `AGENTS.md` verify clause:
      changing a `Protocol` method signature reds every test DOUBLE that
      reimplements it; grep implementors AND `def <method>` stand-ins in one
      pass, and name mypy explicitly in any "green" claim.
- [ ] `ground-steering-text-in-the-real-tool-signatures` -> a test asserting
      every backticked `tool_name(` in the steering preambles resolves to an
      `@mcp.tool()` def.
- [ ] `probe-runtime-on-target-host-early` -> a spike/plan clause: probe a
      dependency's real behavior on the real host before generalizing a design
      across tools.
- [ ] `format-before-the-check-gate` + `format-only-the-files-you-edited-not-whole-dirs`
      -> ONE `AGENTS.md` clause, since they are two halves of the same rule: run
      the WRITING formatter, scoped to the files you edited, before invoking the
      combined gate. A split generator's hand-wrapped imports are the reliable
      trigger.
- [ ] `nix-flake-check-sees-only-tracked-files` +
      `flake-cant-see-untracked-new-files` -> ONE `AGENTS.md` clause: `git add`
      every new file, SOURCE AND TASK RECORD, before `nix flake check` or
      `nix build`.
- [ ] Mark each promoted entry in `LESSONS.md` and re-run
      `tatr check --ledger LESSONS.md`.

## Definition of Done

- Each of the nine lessons above is expressed as a rule in `AGENTS.md`, a
  skill, or a test - not as prose in `LESSONS.md` alone
  (cmd: `rg -n "git show <base>|files you edited|python -m scufris" AGENTS.md`).
- The ledger records every one as PROMOTE against this task, and
  `tatr check --ledger LESSONS.md` exits 0
  (cmd: `tatr check --ledger LESSONS.md`).
- No existing `AGENTS.md` rule is duplicated or contradicted by a new clause
  (manual: read the surrounding section before adding each clause).

## Notes

- Raised while closing 20260731-171431, where three of these
  (`run-the-check-against-the-pre-move-file-before-recording-a-cause`,
  `format-only-the-files-you-edited-not-whole-dirs` and
  `format-before-the-check-gate`) were paid for again, all exactly as written.
  Two lessons reached x3 in that task and are folded in above as pairs with the
  entries they overlap.
- The other pending entries were dispositioned at the same time and are NOT this
  task's work: `orchestrator-steering-is-one-block-two-clauses` and
  `type-change-fails-strict-tsc-not-vitest` are ABSORBED by existing guards,
  `optional-trailing-param-silently-dropped-by-structural-impls` is ABSORBED into
  `protocol-signature-change-hits-the-doubles`, and
  `render-rewrite-orphans-its-css`, `probe-the-stateful-path-not-the-one-shot`
  and the `.env` half of `isolate-state_dir-in-tests-that-assert-config` are
  DEFERRED.
