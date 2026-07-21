# Retro: F0 agents UI polish

- TASK: 20260721-112428
- BRANCH: fix/agents-ui-polish (landed 1c37cd8)
- REVIEW ROUNDS: 1 out-of-context APPROVE (1 NIT accepted)

## What went well

- The out-of-context reviewer verified exactly the async edges I was worried
  about (reconnect loop, double-open, selection race, form-wipe) and they held.
- Gating the SSE reattach on `isActive && events === null` cleanly avoided the
  idle-agent 404/auto-reconnect trap without any extra state.

## What went wrong

- The obvious "add a status interval" would have wiped the create form on every
  tick (the pure full-re-render pattern rebuilds the form and loses input). Not a
  bug that shipped, but it took recognizing that the re-render + a periodic poll
  are in tension. The focus-guard is a workaround; the real fix is F3 moving
  status onto its own page with no form to wipe.

## What to improve next time

- A periodic poll on top of a full-re-render page is a smell - either move the
  volatile data to its own view (coming in F3) or do targeted DOM updates.
  Recognize the tension before adding the interval, not after.

## Action items

- [x] NIT (uncleared interval) accepted for a page-lifetime SPA.
- No new ledger entry: the re-render-vs-poll tension is captured here and is
  resolved structurally by F3.
