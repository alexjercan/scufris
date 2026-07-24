# Retro: idle-unbounded steer/turn timeouts (orchestrator + opencode)

- TASK: 20260724-081804
- BRANCH: bug/steer-idle-timeout
- REVIEW ROUNDS: 1 (APPROVE)

## What went well

- Tracing the actual call shape before coding changed the fix: the plan said
  "make it per-read idle", but reading `send_message` / `_api_call` showed both
  are SYNCHRONOUS single requests where a read timeout IS the total-turn cap. So
  the correct model was `read=None` + an out-of-band backstop, not a per-read
  bound. Coding to the plan verbatim would have shipped a non-fix.
- Tests assert at the request boundary (`request.extensions["timeout"]`) rather
  than poking internals, so they pin observable behavior and fail pre-fix.
- The out-of-context reviewer earned its keep: it found the one entry point
  (`scufris chat` one-shot CLI) where `read=None` had NO backstop - a real
  hang-forever regression this branch introduced, invisible from inside the
  dashboard-path reasoning.

## What went wrong

- I shipped `read=None` reasoning only about the supervised dashboard path and
  did not enumerate the OTHER callers of `backend.stream`. Root cause: when a
  change's safety depends on an out-of-band backstop (here the supervisor
  heartbeat), I reasoned about the backstop for the path in front of me instead
  of asking "does EVERY entry point have this backstop?". The CLI path
  (`_chat_once`) drives the stream directly, with none.

## What to improve next time

- When a fix's safety rests on an out-of-band guard (a supervisor, a timeout one
  layer up), grep every caller of the guarded function and confirm each runs
  under that guard - a guarantee that holds on the main path can be absent on a
  CLI/one-shot/test path.

## Action items

- [x] R1.1 addressed on-branch: `_chat_once` no-output backstop + test.
- No new follow-ups. The retry spike (20260724-081811) remains the only open
  item in this area and was filed at plan time.
