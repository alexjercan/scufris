# Retro: Establish the file-size guard and sweep comment bloat

- TASK: 20260731-171420
- BRANCH: chore/file-size-guard
- REVIEW ROUNDS: 2

## What went well

- The guard was falsified against the REAL tree in both directions before it
  was trusted - drop an allowlist entry and watch it flag `scufris/app.py`, add
  a small file and watch it report a stale entry. That is what turned "the
  check passes" into "the check discriminates", and it is also what caught the
  stale `3769` figure at round 1 (the re-run printed `3764`).
- The 93-citation sweep was driven off one `rg -B6 -A4` dump read in three
  chunks, then applied as exact-match edits. Every citation was accounted for
  and the closing grep was clean on the first try. Never truncating the
  checklist grep is what made the count trustworthy.
- DECISION.md had already settled the three questions that would otherwise have
  been relitigated mid-implementation (`.css` coverage, path-only allowlist,
  Markdown exemption). Nothing in the sweep needed a fresh decision.

## What went wrong

- **Two figures in the close-out came from memory, not from a run.** The
  Evidence section recorded 883 pytest cases (the real number was 895), the
  pre-sweep falsification output `3769`, and `ruff format --check .` as clean
  when it exits 1 on 17 files. Both round-1 MAJORs were this one failure.
  The decision that seemed sound: the numbers had all been observed at SOME
  point in the session, so writing them down felt like transcription rather
  than assertion. It is not - the tree moved underneath two of them (the sweep
  removed 5 lines from `app.py`), and the third was never observed at all,
  because a chained `ruff check . && ruff format --check . ; mypy .` was read
  through `tail -25` and only the tail was seen.
- **The guard's skip rule was written for directories and applied to every
  path component**, so a source file named `result-view.ts` would have been
  silently exempt from the cap. It seemed sound because the strings being
  matched (`__pycache__`, `node_modules`, `result`) are all directory names, so
  the domain felt implied by the values. It is not implied by the code: a
  basename crosses the same `split("/")` as a directory does. A guard that
  silently exempts a file is worse than no guard.
- **The sweep deleted a citation and orphaned the text that referenced it.**
  `scufris/agent.py` kept "the fix for R1.3 stripped codex's environment ...
  (review round 2, R2.1)" after the record link 11 lines above was removed.
  Deleting a reference is only half the edit; whatever pointed at it is now
  dangling and is the same class of lore.
- **The doc-surface sweep missed the two places that enumerate the gate's
  checks** (`README.md` and `.github/workflows/ci.yaml`). Adding a check to
  `flake.nix` is a change to a documented list, not just to a Nix attrset.

## What to improve next time

- Re-run every command whose output a record quotes, at close-out time, and
  paste from that run. A figure observed mid-task has an expiry date.
- Never read a chained verification through `tail`. Run each command bare, or
  echo its exit code per command, so a silent non-zero cannot be reported as
  clean.
- When adding a path-exclusion rule, name the domain in the code
  (`split("/")[:-1]`), not only in the comment, and pin it with a test whose
  fixture is a legitimate file the rule would wrongly match.
- After a deletion sweep, grep for what referenced the deleted thing.
- Adding a `checks` entry to `flake.nix` owes an update to `README.md` and the
  CI workflow's job name and comments in the same task.

## Action items

- 20260731-175511 (filed during this task): bump the pinned tatr from 0.1.0 to
  0.2.0 and disposition the 11 LESSONS.md ledger findings that version
  surfaces. Until it lands, `nix flake check` is red for the duration of every
  task, because the 0.1.0 record parser cannot read the v2 `PLAN STATUS` field.
- R2.1 and R2.2 are open NITs on this branch, accepted with the APPROVE: an
  unscoped 88-column claim in the Evidence section (three `AGENTS.md` table
  rows exceed it, and a Markdown table row cannot be wrapped), and a ragged
  comment paragraph in `.github/workflows/ci.yaml`.
- Ledger housekeeping candidate for a later `/lessons`:
  `nix-flake-check-sees-only-tracked-files` and
  `flake-cant-see-untracked-new-files` are the same root cause recorded twice
  under different commands. Merging them would put the entry at x3 and into
  Pending promotions. Not merged here - that is the ledger owner's call.
- The five split children (171428-171432) each delete their own allowlist
  entries. A stale entry now FAILS the guard, so a child that splits a file
  without pruning its entry cannot land green.
