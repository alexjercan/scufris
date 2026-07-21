# Retro: backend-switch clears session (claude resume bug)

- TASK: 20260721-152034
- BRANCH: fix/backend-switch-clears-session
- REVIEW ROUNDS: 1 (out-of-context APPROVE, zero findings)

## What went well

- Diagnostic-first paid off decisively. Three cheap live probes (a plain claude
  turn, a full same-backend two-turn resume through the real ClaudeBackend, and
  `--resume <unknown-uuid>`) isolated the mechanism in minutes: the base call and
  the happy resume both WORK; only resuming an unknown session yields
  `error_during_execution`. That killed the tempting-but-wrong "claude-opus-4-8
  is an invalid model" theory (the backend never even passes --model).
- The two-layer fix maps cleanly onto the two real failure surfaces: the STORE
  (a backend switch shouldn't carry a session) and the BACKEND (never resume a
  session that isn't there). Each layer got a regression test that fails without
  it, and the store fix benefits codex too, not just claude.
- Factoring `_claude_stream_args` as a pure function made the resume-guard
  unit-testable without running claude at all - the whole fix is covered by fast
  tests plus one end-to-end live confirmation.

## What went wrong

- The reported symptom "errorduringexecution" was doubly obscured: (a) stderr was
  DEVNULL'd in ClaudeBackend, so the real "No conversation found" message never
  surfaced, and (b) the F4 chat renders the error through markdown, so
  `error_during_execution` -> italic `_during_` -> "errorduringexecution". Both
  are papercuts that cost diagnosis time. The stderr one especially: swallowing
  a subprocess's stderr hides exactly the message you need when it fails.
- The user attributed it to the model (claude-opus-4-8), which was a red herring
  - a reminder to reproduce the mechanism rather than trust the reported cause.

## What to improve next time

- When a subprocess turn can fail, capture stderr somewhere retrievable (a log
  at debug level) instead of DEVNULL - the failure message is the diagnosis.
  (Filed as an observation; the DEVNULL was deliberate for pipe-buffer safety,
  so the fix is "tee to a log", not "PIPE and risk deadlock".)
- Render chat error frames as PLAIN TEXT, not markdown, so a backend error is
  legible (noted in TASK.md for a small follow-up F-task).
- Lesson ledgered: probe the app's STATEFUL path (resume/session), not just the
  one-shot invocation, when a tool "works standalone but fails in the app".

## Action items

- [x] Review APPROVE, no follow-ups required for the fix itself.
- [ ] Small follow-up (noted, not filed): F4 chat should render error frames as
      plain text; and consider tee-ing claude stderr to a debug log. Roll into
      F5/F6 or a tiny task if it recurs.
