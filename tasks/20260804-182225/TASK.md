# Carve packages/telegram out of the root distribution

- PRIORITY: 92
- TAGS: feature, v0.2.0, lane2, telegram, packaging
- KIND: TASK
- ACTIVITY: UNDERSTANDING
- GATES: -
- RESOLUTION: -
- PARENT: 20260801-154211
- DEPENDS ON: 20260804-182224

## Story

As the maintainer, I want Telegram to be a workspace package like every other
surviving component, so that the carve epic's ten-unit table is true and the
channel's dependencies are declared rather than implied by living at the root.

The carve epic lists this unit and no task builds it. It is the last member the
epic named and never minted.

## Steps

- [ ] Move `scufris/telegram/` to `packages/telegram/src/scufris_telegram/`
      AFTER the reduction, so what moves is the reduced surface rather than the
      orchestrator coupling.
- [ ] Declare the edge in `DECLARED_GRAPH`. The epic's graph says
      `telegram -> core, chat, hostctl, host`; assert what the imports actually
      are rather than copying the plan, since the check is for EQUALITY.
- [ ] Add its example to `OFFLINE` and `EXAMPLES_BY_MEMBER`. The member gate
      goes red the moment the directory exists, exactly as it did for `chat`.
      `examples/telegram_approval.py` exists at the root today and is not on
      `OFFLINE` - decide whether it becomes the member's offline example or
      whether a new one is needed.
- [ ] Keep the bot token and network out of whatever example is claimed. The
      existing `telegram_bot.py` wants a real token and cannot be the proof.

## Definition of Done

- `scufris_telegram` is a workspace member whose real imports match its declared
  edge (cmd: `python -m pytest tests/test_package_boundaries.py`).
- It names an offline example that imports it
  (cmd: `python -m pytest tests/test_examples.py`).
- No module outside the package imports its internals
  (cmd: `python -m pytest tests/test_package_boundaries.py`).

## Notes

- Depends on the reduction. Carving first would move the orchestrator coupling
  into a package and then have to unpick it there.
- Lane 2 of `tasks/20260801-154211/TASK.md`.
