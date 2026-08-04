# isolate test_telegram lifespan tests from .env (_env_file=None)

- PRIORITY: 34
- TAGS: telegram, test, bug
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Problem

`tests/test_telegram.py`'s lifespan tests build a real `Settings(...)` without
disabling `.env` loading, so on a dev box whose `.env` sets
`SCUFRIS_TELEGRAM_BOT_TOKEN` (needed to run the bot) the ambient token leaks in:

- `test_no_bot_without_token` expects no token -> `telegram_task is None`, but the
  `.env` token makes the bot launch, so the task is a running `_FakeBot.run()`.
- That test monkeypatches `scufris.app.TelegramBot = _FakeBot` and (because the
  token leaked) constructs a `_FakeBot`, populating the class-level
  `_FakeBot.instances`; it clears at start but not at end, so the later
  `test_on_reset_clears_session_serialized` then fails its `_FakeBot.instances
  == []` sanity assertion.

Pre-existing since T4 (20260722-222734): the "isolate config tests from .env"
commit (444f627) added `Settings(_env_file=None)` in `test_config.py` but missed
`test_telegram.py`. Only exposed when the suite runs from a checkout that has a
real `.env` (passes from a sprout worktree, which has none) - surfaced at the T5
flow Finish (`tasks/20260722-222739`). Not a T5 regression; T5 only appended
tests after these two.

## Steps

- [x] Add `_env_file=None` to every real `Settings(...)` construction in
      `tests/test_telegram.py` that is meant to test defaults/explicit inputs:
      the `_settings(tmp_path, **kw)` helper, `test_bot_launches_in_process_when_token_set`,
      `test_no_bot_without_token`, and the e2e `test_end_to_end_receive_turn_reply`
      - mirroring the `Settings(_env_file=None)  # type: ignore[call-arg]` pattern
      already used in `tests/test_config.py`.
- [x] Full check suite green from the MAIN checkout (which has `.env`):
      `ruff check .`, `mypy .`, `python -m pytest` exit 0.

## Changes (as built)

Added `_env_file=None  # type: ignore[call-arg]` to the four `Settings(...)`
sites in `tests/test_telegram.py`. Verified by copying the real `.env` (with
`SCUFRIS_TELEGRAM_BOT_TOKEN`) into the sprout worktree: the T4 pre-fix version of
the two tests failed (2 failed), the fixed version passed (29 passed), and the
full suite was green (ruff/mypy/pytest all exit 0) with `.env` present. The
copied `.env` (gitignored) was removed before committing.

## Definition of Done

1. `test_no_bot_without_token` and `test_on_reset_clears_session_serialized` pass
   when the full suite runs from a checkout containing a `.env` with
   `SCUFRIS_TELEGRAM_BOT_TOKEN`. (test: `python -m pytest tests/test_telegram.py`
   from the main checkout)
2. Full check suite green (ruff, mypy, pytest) from the main checkout.
