# Retro: default to app_server; add mock agent

- DATE: 20260720
- VERDICT: shipped

## What went well

- Making `mock` a third `agent_backend` value (not a separate flag) kept the
  mental model simple: one knob, `SCUFRIS_AGENT_BACKEND`, chooses exec /
  app_server / mock. It slots straight into the existing `build_agent` switch.
- The mock doubles as a runnable demo of the whole streaming pipeline (thinking +
  tools + token-by-token markdown) with no codex login - exactly the kind of
  offline exercise the streaming bug hunt kept needing.

## What went wrong / friction

- Flipping `agent_enabled` to `True` broke several tests that leaned on the old
  default-off to mean "disabled" (the `*_503_when_disabled` family, the config
  echo test, sessions/usage-null tests). Fixed by making each disable the agent
  explicitly via a `_settings(..., agent_enabled=False)` param.

## Lesson

- `tests-that-lean-on-a-default-break-when-it-flips` - a test that asserts
  "disabled" behavior while relying on the *default* being disabled is really
  asserting the default, not the behavior. Set the precondition explicitly so the
  test survives a default change and states its own intent.

## Follow-ups

- A tiny UI "backend: mock/app_server" indicator would make it obvious which
  backend is live (carried over from the streaming-UI retro).
