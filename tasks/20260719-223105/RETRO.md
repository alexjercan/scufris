# Retro: multi-line composer with Enter/Shift-Enter

- DATE: 20260720
- VERDICT: shipped

## What went well

- The spike (20260719-223054) had already scoped this precisely - autosize,
  Enter/Shift+Enter, keep the disabled state - so there was no design ambiguity;
  it went straight to `/work`. Front-loaded research paying off.
- Extracting a shared `submit()` for both the form-submit and the Enter keydown
  kept the send logic single-sourced, and made the `input.disabled` guard cover
  every entry point (Enter, button click) in one place.
- The jsdom tests target the actual decision branches (Enter vs Shift+Enter,
  busy, whitespace) and assert `dispatchEvent`'s return value to prove
  `preventDefault()` fired - a real signal, not a proxy.

## What went wrong / friction

- `sprout rm` refused to delete the worktree because the `web/node_modules`
  symlink shows as untracked (the known `symlink-node_modules-into-fresh-worktrees`
  gotcha, from the other end): it blocked cleanup, not just staging. Had to
  `rm -f` the symlink and `git worktree remove --force`. The branch was deleted
  by `sprout rm` before it bailed, so state was half-torn-down.

## Lessons

- `sprout-rm-blocked-by-node_modules-symlink` - a `web/node_modules` symlink
  added into a sprouted worktree makes `sprout rm` fail on "modified or untracked
  files" (it deletes the branch first, then bails on the worktree). Remove the
  symlink before `sprout rm`, or expect to finish with
  `rm -f web/node_modules && git worktree remove --force`. The symlink trick that
  makes `npm run ci` work in a worktree has a cleanup cost.

## Follow-ups

- A disabled/greyed send button during a turn (the composer is disabled, the
  button is not) would be a clearer signal - fold into the affordances/polish
  task 20260719-223111, not its own task.
- The 200px cap is duplicated (`COMPOSER_MAX_HEIGHT` in JS, `max-height` in CSS);
  if a future task adds a design-tokens layer, unify it there.
