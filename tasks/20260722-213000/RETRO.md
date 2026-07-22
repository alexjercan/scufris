# Retro: Settings interactive 'try it' tool runner UI

- TASK: 20260722-213000
- BRANCH: feature/tool-runner-ui
- REVIEW ROUNDS: 1 (APPROVE)

## What went well

- The prior task's retro lesson was applied FORWARD without a reminder: after
  touching the shared `SettingsActions` interface and `common.ts` types, I ran the
  webpack `npm run build` (the real ts-loader type gate) as part of verify, not just
  `vitest` - and updated the two `SettingsActions` fakes in the same pass, so no red
  build was ever discovered. This is `type-change-fails-strict-tsc-not-vitest` being
  prevented instead of re-hit.
- One-round APPROVE with only two NITs. The out-of-context reviewer independently
  ran the full gate and hammered the load-bearing escaping concern (structured JSON
  path, text path, error path) - the one place a UI runner of untrusted host output
  can go wrong - and confirmed all three paths are inert.
- The backend task's contract (`parameters` schema, `ToolRunResult` shape) landed
  first, so the frontend was a straight consume - the split paid off: two focused
  reviews instead of one sprawling one.

## What went wrong

- Nothing of consequence. Two NITs (a clarifying comment on the empty-`structured`
  fallback, and an optional redundant escape-test) - one fixed, one declined with
  reasoning.

## What to improve next time

- Keep doing the build-gate-after-shared-type-change; it is now proven twice
  (skipped-and-lucky last task, applied-and-clean this one). This is why it sits in
  Pending promotions - it wants a hook or an AGENTS.md verify-step line so it does
  not depend on remembering.

## Action items

- [x] Applied the build-gate lesson during verify (no ledger bump - it was
      prevented, not re-hit; the pending-promotion entry already carries the signal).
- No follow-up tasks: the goal's remaining work is the user's manual acceptance at
  Finish.
