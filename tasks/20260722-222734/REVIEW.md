# Review R1: T4 Telegram transport

- VERDICT: APPROVE
- REVIEWER: out-of-context

## Findings

- [ ] R1.1 (MINOR) scufris/app.py:1346 - A backend `StreamError` mid-turn makes
      `_drain_turn` raise `HTTPException(503)`, which is NOT caught in
      `on_message` (only 409 is). It propagates through `poll_once` into
      `run()`'s broad `except Exception`, gets logged, backs off 3s, and loops.
      The offset was already advanced so the update is not re-delivered, but the
      user gets NO reply and no error message for a failed turn. Consider
      catching the 503 (or any non-409 HTTPException) and returning a short
      "the turn failed" reply, symmetric with the 409->busy handling.
- [ ] R1.2 (MINOR) tests/test_telegram.py:226 - The real `on_message` / `on_reset`
      closures in `_start_telegram_bot` (app.py:1329-1359) have ZERO test
      coverage: the lifespan test swaps in `_FakeBot`, and the transport tests
      use a fake `_Recorder`. So the 409->"busy" mapping, the empty-reply
      coalescing (`done.reply.text or "(...)"`), and the `agent_enabled` guard
      are unexercised - a revert of any of them would not fail a test. The task
      defers the real-app e2e to T5, but these three branches are T4 logic; a
      cheap direct unit test of the two closures would make them revert-sensitive.
- [ ] R1.3 (NIT) scufris/telegram.py:33 - `_dispatch` accepts `/reset` (alias of
      `/new`) and `/start` (alias of `/help`), but `HELP_TEXT` lists only `/new`
      and `/help`. Either document the aliases or drop them so the advertised
      command set matches what is handled.

## Verification notes (checked, fine)

- Offset logic is correct: `self._offset = max(self._offset, update_id + 1)` is
  Telegram's confirm semantics (last id + 1), monotonic, and advances past
  ignored/disallowed updates so nothing is re-delivered forever. The
  `offsets == [0, 13]` and `[0, 21]` tests are revert-sensitive.
- Allowlist gate is airtight: `_handle_update` returns before `_dispatch` for a
  chat not in the frozenset, and also for a non-int chat_id or non-str text, so
  a non-allowlisted chat can neither drive a callback nor receive a `sendMessage`
  (asserted by `not send_route.called`, which fails if the guard is removed).
- Command dispatch: `_command_of` strips `@mention` and lower-cases; `/new`,
  `/help` covered by tests and exact-body `sendMessage` assertions.
- Lifespan wiring: the forward reference to `_start_telegram_bot` from
  `_lifespan` is safe - it is resolved at startup time via the enclosing-scope
  closure cell, long after `create_app` binds it. Task is created only when a
  token is set, cancelled and awaited with `suppress(CancelledError)` on
  shutdown before `supervisor.aclose()`; no leaked task, no swallowed non-cancel
  errors. Both launch/no-launch tests are revert-sensitive.
- httpx client ownership: `_owns_client` guards the `aclose()` in `run()`'s
  `finally`, so an injected client is not closed; the owned client always is.
- `run()`'s `except` re-raises `CancelledError` before the broad `except
  Exception`, so cancellation is not swallowed by the back-off path.
- Config `NoDecode` + `_split_chat_ids` correctly handles the delimited string,
  the JSON array, an empty string (-> `[]`), and rejects a non-numeric segment
  via int coercion. All three config tests pass and are revert-sensitive.
- Full `tests/test_telegram.py` + `tests/test_config.py` run green (26 passed).

## Resolution (author, in-session)

All three findings addressed on the branch; verdict was already APPROVE (no
BLOCKER/MAJOR) and the fixes are additive coverage + one small error branch + a
behavior-preserving extraction, so no second out-of-context round.

- R1.1 (MINOR) FIXED: `on_message` now catches any non-409 `HTTPException` AND a
  broad `Exception` around `launch_turn`/`drain_turn`, logging and returning
  "Sorry - that turn failed. Please try again." So a failed turn (a 503 from a
  backend `StreamError`, or any unexpected error) is reported to the user
  instead of silently dropped. Pinned by `test_on_message_reports_turn_error`.
- R1.2 (MINOR) FIXED: the real callbacks were extracted to a module-level
  `build_telegram_callbacks(settings, agents, supervisor, launch_turn,
  drain_turn)` (behavior-preserving), and six direct unit tests now exercise
  every branch with fakes for the internal turn path: happy reply, disabled
  agent, 409->busy, non-409 error->failure line, empty->coalesce, and reset
  serialization (`serialized(ORCHESTRATOR_ID)` + `set_orchestrator_session(None)`).
  Each is revert-sensitive.
- R1.3 (NIT) FIXED: `HELP_TEXT` now reads "/new (or /reset)". `/start` remains
  handled (Telegram's conventional first-contact command) and simply shows the
  help, which is the expected behavior for it, so it is intentionally not listed
  as a separate command.

Re-verified after the fixes: ruff clean, mypy `Success: no issues found in 48
source files`, full `python -m pytest` green.
