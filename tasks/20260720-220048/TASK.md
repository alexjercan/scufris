# pre-commit hook: reject web/node_modules symlink in commits

- PRIORITY: 0
- TAGS: backlog, bug
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As a scufris developer, I want a pre-commit hook that rejects any commit
staging `web/node_modules` (the sprout-worktree symlink), so that the
`symlink-node_modules` footgun stops corrupting branches. It has recurred (x2);
once `git add -A` staged the symlink into a commit and required
`git rm --cached` + amend + manual symlink deletion to recover. Prose lesson
alone has not held.

## Steps

- [x] Add a pre-commit hook (or check-script) that fails if `web/node_modules` is staged.
- [x] Wire it so it runs in both main checkout and sprout worktrees.
- [x] Verify it triggers on a deliberate `git add web/node_modules` and is silent otherwise.
- [x] Document it briefly in AGENTS.md.

## Definition of Done

- Staging `web/node_modules` makes the commit fail with a clear message (manual: attempt a staged commit).
- Normal commits are unaffected (cmd: a normal `git commit` succeeds).

## Notes

- Promotes the x2 `symlink-node_modules` lesson from prose to a guard (tool > prose).
