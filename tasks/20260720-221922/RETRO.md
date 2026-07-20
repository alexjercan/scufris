# Retro: A0 agent runtime foundation (supervisor + event bus)

- TASK: 20260720-221922
- BRANCH: refactor/agent-runtime (landed 443f8b8)
- REVIEW ROUNDS: 2 (out-of-context round 1: 2 MAJOR + 1 MINOR + 1 NIT; in-session round 2: APPROVE)

## What went well

- Reading the runtime before planning paid off: `list_sessions` already took a
  `cwd`, and the `StreamRunner` alias was the real blast-radius risk. Both facts
  shaped the plan away from hollow work (no fake session-cwd helper) and away from
  a protocol-signature change (kept `cwd` keyword-only on the module runners).
- The `mock` backend + module-level stream runners made the supervisor/eventbus
  fully testable with no codex login; every new test discriminates its mechanism
  (checked budget/heartbeat/cap/serialize/reap by construction).
- The out-of-context reviewer earned its keep: it found two real MAJORs I had
  not seen (the synchronous-reservation race and the unbounded run registry).
  Neither was visible from inside the implementing session's assumptions.

## What went wrong

- R1.1 (reset could jump ahead of its own turn): root cause was modelling
  serialization as "the run acquires a lock when it runs" instead of "the run
  claims its slot when it is started". The old request-held `chat_lock` had
  hidden this by holding across the whole turn; moving execution off the request
  removed that incidental guarantee and I did not replace it with an explicit
  one. Fix: a synchronous FIFO reservation taken in `start()`.
- R1.2 (unbounded `_runs`): I focused on the happy path (start a run, relay it)
  and never asked "who removes a finished run?". A registry that only grows is a
  classic leak; should have been designed in from the first draft.

## What to improve next time

- When moving logic OUT of a scope that provided an incidental guarantee (a
  held lock, a request lifetime, a `with` block), enumerate what that scope was
  silently providing and re-establish each explicitly. The disconnect-safety
  and the serialization were both incidental properties of the old request-held
  lock; one I re-established on purpose, one (ordering) I missed.
- Any in-memory registry keyed by an unbounded id (uuid per request) needs its
  reaping policy written in the same commit as its insertion.

## Action items

- [x] Both MAJORs fixed and pinned by discriminating tests before landing.
- Lessons folded into LESSONS.md: `reserve-serialize-slot-synchronously`,
  `bound-any-per-request-registry`, `moving-logic-off-a-scope-drops-its-incidental-guarantees`.
