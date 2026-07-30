# Retro: Add host approvals over Telegram

- TASK: 20260730-104524
- BRANCH: feat/telegram-approvals
- REVIEW ROUNDS: 2 (3 findings: 1 MAJOR, 1 MINOR, 1 NIT)

What shipped is in `NOTES.md`. Process only here.

## What went well

- **The previous task's decision core paid off exactly as intended.** Building this
  surface was mostly deciding what to SHOW; every rule already existed and could
  only be reached one way. The test that compares the web refusal and the Telegram
  refusal sentence-for-sentence is short because there is one place either can come
  from.
- **Reusing the shared renderer without arguing about it.** The ledger entry
  (`share-one-renderer-so-two-surfaces-cannot-drift`) was read before starting, so
  the chat shows the same text as the dashboard and the agent - and when the cap bug
  turned up, the fix stayed inside that one renderer rather than forking a
  Telegram-shaped variant.
- **Probing the renderer with realistic data.** The MAJOR came from feeding it a
  60-line preview, which is what an R3 closure diff looks like. Reading the code
  would not have shown it; the probe printed a 4096-character body ending in the
  trim marker with the undo line gone.

## What went wrong

- **The cap trimmed the wrong end (R1.1).** I applied the right ledger lesson
  (trim after escaping) and never asked which part of the document was load-bearing.
  Root cause: treating a cap as a length problem rather than as a "what must
  survive" problem - on a message whose last two lines are the undo sentence and the
  result.
- **Two harness mistakes cost most of the debugging time**, both from the same
  assumption: that a bot in a test is a thing I construct, rather than the thing the
  app already started. Awaiting dispatch on the test's loop broke the supervised
  apply, and building a second bot split the announcement state from the tap state.
  Neither was a product defect, and both were invisible until an assertion about
  what the operator SAW failed.

## What to improve next time

- When capping or truncating rendered output, list what must survive before choosing
  where to cut - and pin it with data the real system produces (a closure diff, a
  long journal window), not a short fixture.
- In a test that drives an in-process background surface (a bot, a poller, a worker),
  use the instance the app started and its loop. Constructing a parallel one tests a
  different object than production runs.

## Action items

- [x] Lessons ledger: `cap-what-must-survive-not-just-the-length` and
      `drive-the-instance-the-app-started-on-its-own-loop` appended.
- [ ] Not created as a task: the manual phone check, and an out-of-context review
      round - both need mechanisms this session lacks.
