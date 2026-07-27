# Retro - Dedupe tool-call chips in assistant meta line

## What changed and why

The assistant meta line (`messageMeta` in `web/src/agent-chat-view.ts`) rendered
one chip per tool CALL. A polling orchestrator turn calls `agent_status` /
`pending_agents` many times, so the chip row read like
`... agent_status agent_status agent_status pending_agents agent_status ...`.
Added an order-preserving `distinctTools` helper (`[...new Set(names)]`) and used
it in two display paths: the settled chips (`messageMeta`) and the live
streaming status suffix (`paintStatus`'s `ran ...`), so the two agree.

Alternative considered: a `×N` count badge per tool. Rejected - the user asked
explicitly to "keep only unique ones", and the meta line's job is WHICH tools
ran, not how often. Kept scope to the display list only ("first fix only this
part").

## Difficulties

- `npm` is not on PATH outside the Nix dev shell, and a fresh sprout worktree has
  no `node_modules` (gitignored). Had to run `npm ci` then `npm run ci` inside
  `nix develop .#default --command bash -c '...'` against the absolute worktree
  path. First bare `npm run ci` failed with "npm: command not found".

## Feedback for future sessions

- For a `web/` (frontend) task in this repo, the verify step is
  `nix develop .#default --command bash -c 'cd <worktree>/web && npm ci && npm run ci'`.
  A sprout worktree needs `npm ci` first because `node_modules` is not committed.
- `[...new Set(strings)]` is the idiomatic order-preserving dedupe here; Set
  iteration is insertion order per spec, so no manual seen-set loop is needed.
- Both the settled and live tool-name renderers must stay in sync; a shared
  helper (`distinctTools`) is the cheap way to keep them from drifting.
