# T4: Telegram transport (httpx long-poll, chat->orchestrator session, auth allowlist, token config, in-process launch)

- STATUS: OPEN
- PRIORITY: 33
- TAGS: spike,telegram,feature,backend

## Goal

Add a thin async httpx long-poll Telegram client. Run a `getUpdates` loop
against the Bot API (no public webhook), map the single chat to the
orchestrator's session (`agent_store.orchestrator_session_id` /
`set_orchestrator_session`), with `/new` resetting the session and `/help`
listing commands. Gate access by an allowlist of chat ids
(`SCUFRIS_TELEGRAM_ALLOWED_CHAT_IDS`); ignore everyone else. Token via
`SCUFRIS_TELEGRAM_BOT_TOKEN`. Launch as a background asyncio task inside the app
process when a token is configured, calling the orchestrator through the SAME
internal path as `/api/chat/stream` (no self-HTTP for the bot).

## Notes

- Spike: tasks/20260722-221359/SPIKE.md (Q5).
- Depends on: T1 (so the orchestrator has its tools); transport itself can be
  built in parallel with T2/T3.
- New settings fields in `config.py` (env_prefix `SCUFRIS_`): bot token +
  allowed chat ids.
- Reference the old `github.com/alexjercan/scufris-bot` for the command set.
- Test: integration test with a respx-stubbed Bot API + stubbed/mock backend
  driving one receive -> turn -> reply cycle.
- spike-seeded; plan into steps with /plan before /work.

## Flow State

- FLOW STEP: PLANNED
- PLAN STATUS: APPROVED

## Understanding (grounded in the code, 2026-07-26)

The internal orchestrator turn path already exists and needs no HTTP. In
`create_app()` (`scufris/app.py`):

- `_launch_agent_turn(agents.get(ORCHESTRATOR_ID), None, prompt)` -> `(run_id,
  bus)` runs one orchestrator turn on the supervised backend path (app.py:1159),
  and `_drain_turn(bus)` -> `StreamDone` gives the final reply
  (`done.reply.text`, app.py:1289). This is exactly what `POST /api/chat`
  (app.py:1856) does. Both are CLOSURES inside `create_app`, so the bot cannot
  import them - it must be wired from inside `create_app`/`_lifespan` and driven
  through an injected callback.
- `/new` == the reset path: `async with supervisor.serialized(ORCHESTRATOR_ID):
  agents.set_orchestrator_session(None)` (app.py:1943). The single chat maps to
  the ONE orchestrator session already pinned in `agent_store`
  (`orchestrator_session_id`), so the bot holds no per-chat session state - it
  just gates who may drive it and resets on `/new`.
- The app runs under uvicorn (`run_server`, app.py:1988). `_lifespan`
  (app.py:648) is the one startup/shutdown hook and the correct place to
  `asyncio.create_task` the poll loop (before `yield`) and cancel it (after
  `yield`, beside `supervisor.aclose()`).
- `config.py` `Settings` (env_prefix `SCUFRIS_`, `validate_assignment=True`).
  List env fields use `Field(default_factory=list)` + an optional `mode="before"`
  validator that also accepts a delimited string (see `_split_base_dirs`,
  config.py:204). `.env.example` is the discoverable doc and MUST get both new
  vars (lesson `new-config-field-updates-all-its-surfaces`).
- Tests: `respx>=0.21` is a dev dep; `tests/test_mcp_server.py` shows the
  respx `.mock(side_effect=...)` body-assertion style to mirror.

### T4 vs T5 boundary (decision: keep separate, clean seam)

T5's note allows merging "if the two prove inseparable". They do NOT: T4's
`on_message` seam (return final reply text) is exactly where T5 later wraps a
"typing..." action + a tool-summary line, and adds the examples/ script and the
full app+mock-backend e2e. That is additive, no throwaway shim. So:

- T4 (this task): config fields, `scufris/telegram.py` transport, `_lifespan`
  wiring, and transport-level tests (respx Bot API + a FAKE injected
  orchestrator callback) covering poll/offset/auth/commands/reply, plus a small
  lifespan launch/no-launch test.
- T5: rendering polish (typing action + tool-summary), the `examples/` bot
  script, and the receive->turn->reply e2e through the REAL app + mock backend.

## Steps

- [ ] Add settings to `config.py`: `telegram_bot_token: str | None = None` and
      `telegram_allowed_chat_ids: list[int] = Field(default_factory=list)`, with
      a `mode="before"` validator accepting a comma/colon-separated env string
      ("123,456") as well as a JSON array. Document both in `.env.example`.
- [ ] Add `scufris/telegram.py`: a `TelegramBot` thin async httpx long-poll
      client. Constructor takes `token`, `allowed_chat_ids`, and injected
      `on_message(text) -> Awaitable[str]` + `on_reset() -> Awaitable[None]`
      callbacks (plus an optional `api_base` for stubbing). Methods:
      `poll_once()` (one `getUpdates` batch + dispatch, advancing the offset),
      `run()` (loop `poll_once`, back off on transient error, exit cleanly on
      `CancelledError`, close the owned client), `_dispatch(chat_id, text)`
      (`/new` -> on_reset + confirm; `/help` (and `/start`) -> static command
      list; other text -> on_message -> reply), `_send(chat_id, text)`
      (`sendMessage`). Non-allowlisted chats are ignored silently.
- [ ] Wire the bot into `create_app`/`_lifespan` (`app.py`): build `on_message`
      (`_launch_agent_turn` + `_drain_turn`, returning `done.reply.text`; guards
      `agent_enabled`, maps a 409 to a "busy" reply) and `on_reset`
      (`serialized(ORCHESTRATOR_ID)` + `set_orchestrator_session(None)`). When
      `settings.telegram_bot_token` is set, `asyncio.create_task(bot.run())`
      before `yield`; cancel + await (suppressing `CancelledError`) on shutdown.
      Expose the task/bot on `app.state` for tests.
- [ ] `tests/test_telegram.py` (respx Bot API + fake callbacks): happy-path text
      -> on_message -> sendMessage(reply) with exact-body assertion; unauthorized
      chat ignored (no callback, no send); `/new` -> on_reset + confirm; `/help`
      lists commands; offset advances to last_update_id + 1 on the next poll.
- [ ] Lifespan test (`tests/test_app.py` or `test_telegram.py`): with a token
      set the bot task is created (monkeypatched `run`), with no token it is not.
- [ ] `test_config.py`: the two new fields load from `SCUFRIS_` env and the
      allowlist parses both a delimited string and a JSON array to `list[int]`.
- [ ] Run the full check suite green (ruff format changed files; ruff check;
      mypy no NEW errors vs base; `python -m pytest` from the worktree).

## Definition of Done

1. `telegram_bot_token` + `telegram_allowed_chat_ids` load from `SCUFRIS_` env;
   allowlist parses "123,456" and `[123,456]` to `list[int]`.
   (test: `tests/test_config.py`)
2. `TelegramBot.poll_once` long-polls `getUpdates`, dispatches, and advances the
   offset past processed updates. (test: `tests/test_telegram.py` offset)
3. A non-allowlisted chat is ignored: no callback, no `sendMessage`.
   (test: `tests/test_telegram.py` unauthorized)
4. `/new` calls `on_reset` (session reset) and confirms; `/help` lists the
   commands; neither drives `on_message`. (test: `tests/test_telegram.py`)
5. A text message drives `on_message` and replies via `sendMessage` with the
   returned reply text (exact-body respx assertion).
   (test: `tests/test_telegram.py` happy path)
6. The bot launches in-process from `_lifespan` only when a token is configured,
   and is cancelled cleanly on shutdown. (test: lifespan launch/no-launch)
7. `.env.example` documents `SCUFRIS_TELEGRAM_BOT_TOKEN` and
   `SCUFRIS_TELEGRAM_ALLOWED_CHAT_IDS`.
   (cmd: `grep -n TELEGRAM .env.example`)
8. Full check suite green (ruff, mypy adds no new errors vs base, pytest).

Out of scope (T5): "typing..." chat action, tool-summary line, `examples/` bot
script, and the receive->turn->reply e2e through the real app + mock backend.
