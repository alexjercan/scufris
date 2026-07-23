# Retro: Read-only project skills+tools cards on the agent settings page

- TASK: 20260723-225621
- BRANCH: feature/project-capabilities-ui
- REVIEW ROUNDS: 1 (APPROVE, one cosmetic NIT addressed)

## What went well

- Mirroring the existing `agentToolsPanel` and reusing the `panel()` helper for
  empty states kept the new cards native to the page with minimal new surface -
  the reviewer confirmed consistency with repo conventions.
- Test-first at the render altitude: wrote the three vitest cases (populated,
  empty, no-project) against the card text, watched them fail before the render
  functions existed, then implemented. The reviewer independently re-derived
  that the tests fail if the render is removed.
- The backend endpoint's field names/types were mirrored exactly in the TS
  interfaces because I re-read `scufris/project_capabilities.py` in the worktree
  first - zero field-name drift.

## What went wrong

- The task's non-ASCII DoD grep was scoped to the WHOLE file
  (`grep -nP "[^\x00-\x7f]" web/src/agent-settings-view.ts ...`) and self-matched
  two PRE-EXISTING glyphs (the `<-` back-link arrow, a middot in the usage panel)
  that this diff never touched. Root cause: an absence-proving grep written
  against the file instead of the diff, the same self-match failure the plan
  skill already warns about for `tasks/`-tree greps - it applies to "no new
  non-ASCII" too. Cost was only a verification note, not rework, but the DoD
  cmd read as a failure when the intent held.

## What to improve next time

- Scope an absence-proving grep (no new non-ASCII, no stale symbol) to the DIFF,
  not the whole file: `git diff <base>... -- <path> | grep -nP "[^\x00-\x7f]"`,
  or grep only the added lines. Phrase such DoD cmds that way at plan time.

## Action items

- [x] Ledger: added `scope-absence-greps-to-the-diff-not-the-file` (x1).
- [x] NIT R1.1 addressed in-branch (comment on ProjectSkill noting the backend
  default); no follow-up task needed.
