# Retro: F1 SPA dynamic routing + agent-detail shell

- TASK: 20260721-112433
- BRANCH: feature/agent-routing (landed 4f4a8e1)
- REVIEW ROUNDS: 1 out-of-context APPROVE (1 NIT, no change)

## What went well

- The structural gate landed cleanly by leaning on Starlette's route order +
  non-empty path segment: `/agents/{id}` before the static mount, after the /api
  routes, and `/agents/` (empty segment) falls through to the list. No custom
  fallback middleware needed.
- Verifying routing with BOTH an explicit test (list-not-shadowed, api-not-shell)
  AND a live serve caught the class of bug a green unit test misses - the e2e
  confirmed `/agents/<id>` and `/settings` serve the real built shell.

## What went wrong

- Nothing. The one risk (a `{id}` route swallowing the list or a static asset)
  was designed against and pinned.

## What to improve next time

- Keep pairing routing changes with a live serve check - the frontend-verify
  lesson applies double to anything that touches path precedence.

## Action items

- [x] APPROVE; NIT (className escaping) left (server enum).
