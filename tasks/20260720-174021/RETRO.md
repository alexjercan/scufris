# Retro: Fix pre-existing mypy red on master (FakeAgent/LogRecord)

- TASK: 20260720-174021
- BRANCH: bug/mypy-green
- REVIEW ROUNDS: 1 (APPROVE)

## What went well

- Reproduced first: ran `mypy .` on clean master before touching anything,
  which grouped the 18 errors into exactly three root causes (FakeAgent
  return type, LogRecord.req read, image_paths index) and made the fix
  aimed rather than guessed.
- The three fixes were type-honest, not silencers: annotating
  `chat_stream -> AsyncIterator[StreamEvent]` makes FakeAgent structurally
  satisfy the `Agent` protocol at all 11 `create_app` call sites, so mypy
  passing now actually proves the double matches the protocol. The
  out-of-context reviewer independently confirmed no drift was masked.

## What went wrong

- Root cause of the bug: task 20260720-144530 (image attachments) added the
  `image_paths` param and left FakeAgent.chat_stream at
  `AsyncIterator[object]`, then closed claiming a green suite - mypy was
  never actually run there. This is the `protocol-signature-change-hits-the
  -doubles` lesson biting a second time, now as an unnoticed red rather than
  a caught one.
- Small dead end: reached for `getattr(record, "req")` for the dynamic
  attribute; ruff B009 rejects getattr-with-a-constant. Switched to
  `record.__dict__["req"]` (reads Any, no ignore, no B009).
- Process snag: wrote the round-1 verification notes as
  `- [x] R1.1 (verified) ...` checkbox findings, and `tatr check` parses that
  syntax as a finding and rejects "verified" as a severity. Had to reword
  them as plain "Verified:" prose bullets.

## What to improve next time

- A "green suite" claim must name mypy explicitly, not just pytest. mypy
  drift is invisible to a passing pytest run (the tests execute fine; only
  the types are wrong).
- In REVIEW.md, verification notes are not findings - use plain prose
  bullets, and reserve the `- [ ] Rn.n (SEVERITY)` syntax for the four real
  severities so `tatr check` stays green.

## Action items

- [x] Reworded REVIEW.md verification notes to pass `tatr check` (done this
      cycle).
- No follow-up code task: the fix is complete and the suite is green.
