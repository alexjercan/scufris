# Retro: B2 permission modes

- TASK: 20260721-112430
- BRANCH: refactor/permission-modes (landed f54fc89)
- REVIEW ROUNDS: 1 out-of-context APPROVE (zero findings)

## What went well

- Probing the exact codex/claude flag values LIVE (`--help`) before wiring - the
  now-x3 lesson - meant the mode->flag map was right the first time; the reviewer
  confirmed all six mappings.
- Migrating the legacy write_enabled BEFORE model_validate (not after) closed the
  one real trap: a dropped field would otherwise be ignored and a write-enabled
  agent silently become read-only. Pinned by a test.
- A wide (10-file) rename went clean because the field flowed through one seam
  (AgentBackend.stream) with keyword args - the fakes updated in one pass.

## What went wrong

- Nothing. A zero-finding review on a cross-cutting change is the payoff of the
  earlier discipline (verified flags, migrate-first, one seam).

## What to improve next time

- Keep doing the "probe the dependency's flags live before wiring" step for any
  external-CLI mapping - it is now paying off every time.

## Action items

- [x] Clean APPROVE, nothing to address.
- No new ledger entry.
