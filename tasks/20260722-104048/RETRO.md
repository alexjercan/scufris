# Retro: terminal styling pass (kitty palette, full mono, sharp edges)

- TASK: 20260722-104048
- BRANCH: feature/terminal-styling-pass (landed 6d13a08)
- REVIEW ROUNDS: 1 (out-of-context APPROVE, one dead-token NIT wired in)

See TASK.md for the pinned direction + what/why and REVIEW.md for the findings.
Process notes only here.

## What went well

- For a SUBJECTIVE task, I did not guess: I read the kitty config + the current
  palette, found the concrete tension in the ask ("make it like kitty" vs "keep
  the scheme's character"), and used ONE AskUserQuestion round with color/edge
  PREVIEWS to pin four forks (palette / accent / typography / edges) before
  touching CSS. The whole implementation then followed mechanically from the
  answers - no rework, no taste dispute in review.
- The recon surfaced a real bug behind the vague "buttons are weak" complaint:
  `background: var(--bg)` with only `--bg-0/--bg-1` defined -> transparent ghost
  buttons. Naming that in the plan turned an aesthetic ask into a concrete fix.
- I turned an "untestable" CSS task into a tested one: a token-integrity test that
  parses style.css and asserts no `var(--x)` is used without a fallback unless
  defined. It immediately EARNED its keep - it caught a SECOND undefined `var(--bg)`
  on the inputs that I had missed when I only fixed the buttons. That is exactly
  the pending-promotion lesson (styled-but-undefined tokens) made executable.
- The whole restyle stayed in the shared style.css (no HTML/TS churn), because the
  pages already share one class vocabulary - a token-level change propagated.

## What went wrong

- I fixed the `var(--bg)` bug on the buttons but missed the identical
  `var(--bg)` on `.settings__select/.__input` - I fixed the instance I was
  looking at, not the CLASS of the bug. The token test caught it (good), but only
  because I wrote the test; a targeted grep for `var(--bg)` at fix time would have
  found both at once.
- A commit message with backticked tokens (`` `var(--bg)` ``) got mangled by the
  shell (command substitution ate them); I had to amend via a message file. Reflex
  to avoid: never put backticks/`$()` in a `git commit -m` string - use `-F <file>`
  or a heredoc when the message contains shell metacharacters.

## What to improve next time

- When fixing a token/value bug, grep the WHOLE file for that exact token
  (`var(--bg)`) and fix every site in one pass, rather than fixing the occurrence
  under the cursor - the same enumerate-the-class discipline the ledger already
  preaches for CSS orphans and synthetic-collection assertions.
- Commit messages containing code punctuation go through `git commit -F <file>`,
  not `-m "...backticks..."`.

## Action items

- [x] Adopted the NIT: wired `--red-bright` into the danger-button hover.
- [x] Add lesson `no-backticks-in-git-commit-m` (shell eats them).
- [x] The token-integrity test stands as the executable form of the pending
      `render-rewrite-orphans-its-css` promotion (styled-but-undefined tokens);
      noted in that ledger entry's promotion candidate.
