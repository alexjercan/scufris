# Retro: project detail page /projects/<id>

- TASK: 20260722-104043
- BRANCH: feature/project-detail-page (landed 26232e0)
- REVIEW ROUNDS: 1 (out-of-context APPROVE, one dead-CSS NIT adopted)

See TASK.md for what/why (incl. the folded-in spike findings) and REVIEW.md for
the findings. Process notes only here.

## What went well

- The parallel recon agent + my own tatr grep front-loaded the feasibility
  question (can we show tatr tasks per project?) and found the backend was ALMOST
  ENTIRELY already built: `GET /api/projects/{id}` + `/tasks` +
  `read_project_tasks` existed, and `/api/agents` covers agents-per-project by a
  client filter. That turned a "needs backend endpoints" task into a
  mostly-frontend page + two shell routes, and the plan said so up front.
- Mirroring the per-agent detail shell EXACTLY (backend `_detail_shell` + two
  routes before the static mount, webpack entry + plugin + dev rewrite, an
  `idFromPath` helper) made routing correct by construction - the reviewer's main
  risk (does `/projects/{id}` shadow the static `/projects/` list?) was already
  covered by a test copied from the agent-detail test.
- Consolidating the inline detail panel into the page (rather than keeping both)
  kept one detail surface, matching how agents already work (list navigates, no
  inline detail). The removal was clean - no dangling `select`/`remove` refs.

## What went wrong

- The dead-CSS NIT: removing `projects__item--active` (the inline selection's
  highlight) orphaned `.projects__item--active .projects__name`, and I did not
  sweep the CSS for the class I stopped emitting. This is the SAME lesson already
  in the ledger (`render-rewrite-orphans-its-css`) - I removed a state class from
  the TS without grepping the stylesheet for it. Also, turning the name from a
  `<button>` into an `<a>` silently inherited the default link underline until I
  restyled it (caught while adopting the NIT, not before).

## What to improve next time

- When a change STOPS emitting a class/state (here `projects__item--active`), grep
  `web/src/*.css` for that class in the SAME edit and remove the now-dead rules -
  the existing `render-rewrite-orphans-its-css` discipline applies to removing a
  state class, not just rewriting a render.
- When changing an element's TAG (button -> anchor, div -> anchor), immediately
  check the shared class's CSS for tag-default assumptions (anchor underline/color)
  - the U5 wordmark taught this once; it recurred here on `.projects__name`.

## Action items

- [x] Adopted the NIT: removed the dead `.projects__item--active` rule; styled the
      now-anchor `.projects__name` (no underline + cyan hover).
- [x] Bumped `render-rewrite-orphans-its-css` (now covers "stop emitting a state
      class" too).
