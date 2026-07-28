# Retro: Cancel in-flight chat runs (stop button + cancel_agent tool + CANCELLED)

- TASK: 20260728-134840
- BRANCH: feature/chat-cancel
- REVIEW ROUNDS: 1 (APPROVE)

See `TASK.md` close-out + `NOTES.md` for what/why; this is process only.

## What went well

- Exploration-first paid off. Two parallel Explore agents mapped the run
  lifecycle (supervisor task -> _drain aclose -> persist callback) and the chat
  UI before any code, so the cancel signal had one obvious injection point and
  the whole feature was additive - `agent_store.py` needed zero change because
  the map already showed `pending_outcomes`/`wake.py` only act on
  WAITING/REPORTED/ERROR.
- Reading the area lessons up front directly shaped the tests: the async-httpx
  held-open-run pattern, "assert the durable record not /status", and async
  cancel endpoint all came straight from `LESSONS.md` and avoided a deadlocked
  or non-converging test.
- One clean review round (APPROVE, two non-blocking findings) - the out-of-context
  reviewer re-ran the full suite and re-derived the aclose-proof, confirming the
  test genuinely fails if the cleanup is removed.

## What went wrong

- R1.1 (orchestrator `forkTurn` dropped the new `signal` param). Root cause: I
  threaded the new optional `signal` through the two STREAMING call sites
  (`streamTurn`, per-agent `forkTurn`) but the orchestrator's `forkTurn` is a
  bespoke non-streaming `fetch`, and TS structural typing lets an implementer
  omit a trailing optional param with NO compile error - so mypy/tsc/webpack all
  stayed green and the gap only showed under a skeptical read of every
  implementer.
- Minor DoD churn: planned the endpoint as `204` then chose `200 {cancelled}`
  while implementing (mirroring the sibling endpoints). Harmless, but it meant
  reconciling the DoD after approval instead of pinning the contract at plan time.

## What to improve next time

- When adding an optional param to a SHARED injected-config/callback interface,
  grep every implementer of that interface and thread it through each - the
  compiler will not flag the ones that silently drop it. Do not assume the
  streaming call sites are the only ones.
- When the mechanism is already well understood at plan time, pin the response
  contract (status code + body shape) in the DoD then, not during implementation.

## Action items

- [x] Ledger: add `optional-trailing-param-silently-dropped-by-structural-impls`
      (variant of `protocol-signature-change-hits-the-doubles`).
- [x] R1.1 + R1.2 fixed on the branch (see REVIEW.md).
- No follow-up code tasks: the feature is complete and the manual DoD item is the
  only open acceptance check.
