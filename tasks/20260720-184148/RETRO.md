# Retro: Settings UI - interactive config controls + tools editing

- TASK: 20260720-184148
- BRANCH: feature/settings-ui-controls
- REVIEW ROUNDS: 2 (R1 REQUEST_CHANGES -> add endpoint tests -> R2 APPROVE)

## What went well

- The `SettingsActions` injection seam kept `renderSettings` pure and
  jsdom-testable while wiring real fetches only in `startSettings`, and gave a
  clean single-authoritative-render (reload-after-mutation, no client copy).
- Spotted the whole-list-PATCH limitation early (the config view exposes only
  `{id, source}`, so the client can't rebuild specs) and added incremental
  POST/DELETE endpoints - the correct inseparable-slice call rather than a
  shim.
- Verified end to end by serving the built bundle through uvicorn and curling
  every new route, not just trusting the green build.

## What went wrong

- `git add -A` staged the `web/node_modules` symlink into the commit - the
  exact thing LESSONS.md `symlink-node_modules-into-fresh-worktrees` warns
  against. Caught it in the commit stat, `git rm --cached` + amend + removed the
  symlink. Root cause: reflex `git add -A` in a worktree with the symlink.
- The shared-type-change (`enabled` on AgentTool, `writable` on AgentConfig)
  passed vitest but failed the webpack build on `agent-view.test.ts`'s factory
  - `type-change-fails-strict-tsc` again. Should have grepped every literal of
  those types across web/src before the first `npm run ci`.
- Review R1.1 (MAJOR): the net-new POST/DELETE endpoints had no direct tests -
  I'd tested the reused PATCH path and assumed coverage. The reviewer correctly
  refused: a new route's branches (409/404/403/422) are unverified until tested
  directly.

## What to improve next time

- In a sprout worktree, NEVER `git add -A`; stage explicit paths (the symlink
  never matches `.gitignore`'s `node_modules/` and slips in).
- After a shared-type change, grep every constructor of the type across the
  whole frontend BEFORE running ci (the lesson says this literally).
- When a task adds NEW endpoints beside a reused one, write tests for the new
  routes' own branches - reused-path tests do not cover them.

## Action items

- [x] Bumped `symlink-node_modules-into-fresh-worktrees` (recurred via git
      add -A) and added `test-the-net-new-route-not-the-reused-path` to
      LESSONS.md.
- No follow-up code task.
