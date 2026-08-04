# Understanding: carve packages/telegram

## What changes

The tenth workspace member. `scufris/telegram/` becomes
`packages/telegram/src/scufris_telegram/`.

## Surfaces

- `scufris/telegram/` - ten modules, moving.
- `tests/test_package_boundaries.py` - `DECLARED_GRAPH` gains an entry, and the
  root `scufris` entry may LOSE one if nothing at the root imports Telegram
  after the move.
- `tests/test_examples.py` - `OFFLINE` and `EXAMPLES_BY_MEMBER`.
- `pyproject.toml` - `[tool.uv.sources]`, if the root depends on it.

## Data and interfaces

The epic's declared graph says `telegram -> core, chat, hostctl, host`. That is
a PLAN, and `DECLARED_GRAPH` is checked for EQUALITY - so the entry must state
what the imports actually are after the reduction, not what the epic predicted.
Copying the plan is the failure mode the equality check exists to catch.

## Sketches

```
  the member gate fires on directory creation, exactly as it did for chat:

    packages/telegram/src/scufris_telegram/  exists
              |
              +--> _import_roots() globs packages/*/src/*
              +--> EXAMPLES_BY_MEMBER demands an offline example
              +--> DECLARED_GRAPH demands an exact edge set

  so the first commit is: move + graph entry + example registration,
  and only then anything else
```

## Shape

Reduction first, then carve. Carving first would move the orchestrator coupling
into a package and then require unpicking it there, which is strictly more work
in a worse place.

## Consequences and open questions

- **The example is the real question.** `examples/telegram_approval.py` and
  `examples/telegram_bot.py` both exist and NEITHER is on `OFFLINE` -
  `test_examples.py`'s own docstring names `telegram_bot.py` as wanting a token.
  So the member gate cannot be satisfied by what is there. Either
  `telegram_approval.py` is made genuinely offline with a fake bot, or a new
  example is written. This is unglamorous and it is the task's main cost.
- **Open:** whether the root still imports Telegram after the move. If the
  composition root wires it, the `scufris -> scufris_telegram` edge is declared
  here; if the module registry does it in Lane 6, this task must NOT declare it,
  because the check is for equality.
- **Open:** whether `contracts.py` (`ApprovalOps`, `ApprovalOutcome`,
  `OrchestratorInfo`, `SettingsOps`) survives the reduction intact. It is the
  seam the package talks to the rest of the app through, and it names the
  orchestrator. Read it before assuming the move is mechanical.
