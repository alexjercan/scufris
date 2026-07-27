# Retro: Match sub-agent Tools section to orchestrator tool cards

## What went well

- The fix was a reuse, not a rewrite: the orchestrator already had the exact
  read-only card renderer (`toolCard`). Exporting it and dropping it into
  `agentToolsPanel` made the two surfaces pixel-identical AND unable to drift,
  in ~10 net lines. The read-only guarantee came for free because the toggle and
  runner live in wrappers (`toolControlCard`/`renderToolControls`), not in
  `toolCard` itself - so "bare `toolCard` = read-only" is structural, not a
  convention someone has to remember.
- Exploring both render paths up front (data shape, CSS classes, which wrapper
  adds the interactive bits) meant the plan named the concrete artifact - a
  `tool-grid` of bare `toolCard`s - with no mid-build surprises.

## What went wrong / difficulties

- Nothing substantive. Two mechanical papercuts: (1) the Write tool requires a
  prior Read of the exact path, and the worktree copy is a different path than
  the main-checkout file I had already read, so the first edits to each worktree
  file bounced until I re-Read them there. (2) `tatr new` wrote the task onto the
  main checkout before sprouting, so I had to carry-and-clean the folder into the
  worktree.

## What to do differently next time

- When a change is "make surface A look like surface B" and B already has a
  self-contained renderer, the default move is export-and-reuse B's renderer, not
  restyle A's markup. Check for the shared component BEFORE writing any CSS.
- After sprouting, Read a file in the worktree path before the first Edit there,
  even if an identical copy was already read in the main checkout - the harness
  tracks read-state per path.

## Lesson candidates (for /lessons)

- `reuse-the-existing-read-only-renderer-to-match-two-surfaces`: to make two UI
  surfaces look identical, export the smaller/read-only render fn one already has
  and reuse it, rather than duplicating markup. Structural read-only guarantees
  (interactive bits added by a wrapper, not the base fn) then carry over for free
  and the surfaces cannot drift. Here: `toolCard` shared between the orchestrator
  console and the sub-agent Tools grid. 20260727-101518.
