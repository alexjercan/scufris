# Retro: Orchestrator permission mode - default auto + expose in settings

- TASK: 20260723-001243
- BRANCH: feature/orchestrator-auto-default
- REVIEW ROUNDS: 1 (APPROVE; 1 MINOR fixed in-session)

See TASK.md for what changed and why; this is process only.

## What went well

- Grounding at filing time paid off: the task body (written the night before from a
  code read) had already established the mode was settings-derived and writable, so
  the "expose in settings" half resolved to zero frontend work once verified.
- Red-first discipline: all three new tests were watched fail (`manual != auto`)
  before the one-line flip; and grepping for tests that assumed the old default
  BEFORE flipping meant zero surprise failures (the manual assertions all concern
  regular agents, unaffected).
- The out-of-context reviewer earned its keep again: it swept every reader of
  `agent_permission_mode` and found the one surface the session missed - the CLI
  one-shot chat ignored the setting entirely (pre-existing, but the new docs made
  it a lie). Fixed as "one orchestrator, one posture" + a kwargs-recording test.

## What went wrong

- R1.1: I verified the dashboard path end-to-end but did not enumerate ALL callers
  of `backend.stream` for posture consistency - the CLI path passed
  `is_orchestrator=True` but no `permission_mode`. Root cause: I treated the T1-era
  call-site list as "checked" for a different property (orchestrator flag) and did
  not re-sweep it for the NEW property (mode). A property change should re-sweep
  the same call-site list it rides through.

## What to improve next time

- When adding/changing a per-turn property, grep every `backend.stream(` call site
  for THAT property, even ones audited before for other reasons - each new
  property needs its own pass over the same seam.

## Action items

- None filed: one-off process note (first occurrence), captured here. The Telegram
  T4 auth-allowlist note stands as the future gate for the auto posture.
