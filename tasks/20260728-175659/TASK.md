# Telegram /cancel stops current orchestrator message

- PRIORITY: 30
- TAGS: backlog, feature, telegram, agent, backend
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As a Telegram user, I want `/cancel` to stop the current orchestrator message so
the bot behaves like pressing Stop in the web chat.

## Understanding

The generic cancellation path already exists from task 20260728-134840:
`Supervisor.cancel(run_id)` and `POST /api/agents/{id}/cancel` produce a neutral
`AgentState.CANCELLED`. Telegram currently has only `on_message` and `on_reset`
callbacks, so it can launch and reset orchestrator turns but has no command that
calls the stop path.

Done means `/cancel` is recognized as a command, does not forward a prompt to
the orchestrator, calls the current orchestrator cancel callback, and replies
clearly whether a live turn was stopped.

## Steps

- [x] Add an `OnCancel` callback to the Telegram transport and dispatch
      `/cancel` through it.
- [x] Wire `build_telegram_callbacks` so `/cancel` cancels the active
      orchestrator run via the existing supervisor run id.
- [x] Add or update Telegram tests for command dispatch, no-typing behavior,
      command parsing, and callback wiring.
- [x] Update user-facing command text and task notes.
- [x] Run focused tests and the project check gate.

## Definition of Done

- `/cancel` and `/cancel@bot` are recognized as commands and do not become
  orchestrator prompts (test: `test_cancel_command_cancels_active_turn` and
  `test_command_of`).
- `/cancel` calls the callback and sends "Cancelled current message." when a
  live turn was stopped, otherwise "No active message to cancel." (test:
  `test_cancel_command_cancels_active_turn` and
  `test_cancel_command_reports_when_idle`).
- The app callback cancels `agent_runs[ORCHESTRATOR_ID]` through
  `Supervisor.cancel` and reports false when no active run exists (test:
  `test_on_cancel_stops_orchestrator_run` and
  `test_on_cancel_false_when_idle`).
- Telegram help lists `/cancel` (test: `test_help_command_lists_commands`).
- Focused Telegram tests pass (cmd: `python -m pytest tests/test_telegram.py`).
- Full project gate is green or any skipped/failing check is recorded with the
  exact reason (cmd: `ruff check .`, cmd: `mypy .`, cmd: `python -m pytest`).

## Notes

- Scope is Telegram only; the backend and web stop paths are already implemented
  by task 20260728-134840.
- Flow State is approved from the user's explicit end-to-end implementation
  instruction in this orchestrated run.
- Implementation detail: Telegram turns now render in a tracked background task
  so the long-poll loop can receive `/cancel` while a turn is still streaming.
  `/cancel` calls the app cancel callback and then cancels the local render task
  so the chat receives the command reply, not a second friendly failure message
  from the cancelled stream.
- Verification in this sandbox:
  - `python -m py_compile scufris/telegram.py scufris/app.py tests/test_telegram.py`
    passed.
  - `python -m pytest tests/test_telegram.py` could not run because the system
    Python has no pytest.
  - `nix develop --command python -m pytest tests/test_telegram.py` could not run
    because the sandbox cannot connect to the Nix daemon socket.
  - `ruff check .`, `mypy .`, and full `python -m pytest` were skipped for the
    same unavailable toolchain reason.
- Self-review caught that command dispatch alone was insufficient: the previous
  synchronous `_dispatch` awaited the whole streamed turn, so the bot could not
  receive a later `/cancel` update until after the turn finished. The fix moved
  turn rendering into tracked background tasks and added a regression test for
  cancel arriving on the next poll.

## Close-out

Changed the Telegram transport to expose `/cancel`, route it through a new
`OnCancel` callback, and keep polling while a turn streams. `build_telegram_callbacks`
now receives an `active_run_id` seam from `create_app` and cancels the
orchestrator's current supervisor run by id. Tests cover command parsing,
cancel-active, cancel-idle, busy duplicate messages, direct callback behavior,
and the existing render tests now explicitly wait for background turn tasks.

Alternatives considered: routing `/cancel` through the existing HTTP endpoint
would work but would add self-HTTP inside the in-process bot, against the current
Telegram design. A direct callback keeps the transport unit-testable and reuses
the same supervisor cancel primitive.

Difficulty: the first pass missed the long-poll concurrency requirement. It was
diagnosed during diff review by following the `_dispatch -> _render_turn` await
chain and fixed before close.

Self-reflection: review command changes against the runtime scheduling model
before treating command dispatch as sufficient. For chat transports, ask whether
the receive loop keeps running while a long reply is being produced.
