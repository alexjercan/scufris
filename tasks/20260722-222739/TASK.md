# T5: Telegram reply rendering + end-to-end example (final-per-turn + tool summary; examples/ script; respx integration test)

- STATUS: CLOSED
- PRIORITY: 32
- TAGS: spike,telegram,feature,ui
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Goal

Reply rendering plus an end-to-end proof for the Telegram bot. Render one final
message per turn with a "typing..." chat action while the orchestrator streams,
and a short tool-summary line (full edited-message token streaming is a later
polish). Ship an `examples/` script that boots the bot against a stubbed Bot
API + the mock backend, and an integration test exercising receive-message ->
orchestrator turn -> reply.

## Notes

- Spike: tasks/20260722-221359/SPIKE.md (Q5, rendering).
- Depends on: T4. May merge with T4 at plan time if the two prove inseparable
  (flow allows re-cutting inseparable tasks).
- Harness-first (AGENTS.md): respx-stubbed Telegram + mock backend; the
  `examples/` script doubles as documentation.
- spike-seeded; plan into steps with /plan before /work.

## Understanding (grounded in the code, 2026-07-26)

T4 (CLOSED) shipped the transport: `scufris/telegram.py` `TelegramBot` (long-poll
`getUpdates`, allowlist auth, `/new`|`/help` dispatch, `on_message`/`on_reset`
callbacks, one-message-per-turn reply), plus `build_telegram_callbacks` and
`_start_telegram_bot` wiring in `app.py` and transport tests. T5 is the additive
polish on top of that seam - NO throwaway shim (the T4 task confirmed the two are
separable at the `on_message -> reply text` seam).

Three deliverables, each landing on an existing, verified seam:

1. **Tool-summary line.** The final reply already carries the turn's tool calls:
   `StreamDone.reply.tool_calls: list[ToolCall]` (`ToolCall(server, tool, status)`
   in `sessions.py`). `_launch_agent_turn`/`_drain_turn` pass `done.reply` through
   untouched, so `build_telegram_callbacks.on_message` (`app.py:621`) can render a
   short footer from `done.reply.tool_calls`. The MockBackend leaves `tool_calls`
   empty; codex populates it. Rendering must be a pure, unit-testable helper in
   `telegram.py`, ASCII-only (AGENTS.md: no emoji/typographic chars).

2. **"typing..." chat action.** The bot must show `sendChatAction(action=typing)`
   while a turn runs. This is a pure transport concern in `TelegramBot._dispatch`:
   only the `on_message` branch (a real turn) needs it; `/new`|`/help` reply
   instantly. Telegram's typing status lasts ~5s, so a keepalive re-sends it while
   the turn is in flight. Send one action immediately (so even a fast turn shows
   typing, and the test is deterministic), then a keepalive loop for long turns,
   cancelled in a `finally` before the reply is sent.

3. **examples/ script + e2e test.** `examples/telegram_bot.py` mirrors
   `examples/comms_loop.py`'s shape: boot the REAL app in-process against the mock
   backend, stub the Bot API with respx, feed one text update, print the rendered
   reply + observed typing action, exit 0/1. To actually demonstrate the
   tool-summary, the example (and the e2e) override `MockBackend.stream` to emit a
   couple of `StreamTool` events + a reply carrying `tool_calls` (the test-proven
   pattern at `test_app.py:2699`). The e2e (`tests/test_telegram.py`) runs the real
   `_lifespan` (`app.router.lifespan_context(app)`) so the production
   `_start_telegram_bot` task drives the loop, stubs getUpdates (one update then
   empty) + sendMessage + sendChatAction, and bounded-waits for the captured send.

Seam/verification notes grounded in the code + lessons:
- Keep `OnMessage = Callable[[str], Awaitable[str]]` unchanged (T4 seam);
  `on_message` returns the FULLY RENDERED string (reply text + footer), so the bot
  stays display-agnostic and existing `on_message` tests (empty `tool_calls` ->
  no footer) keep passing. `render_reply("", [])` -> "" still coalesces to the
  "(no text)" line; `render_reply("", [tools])` -> footer-only (non-empty).
- Adding the typing action makes the on_message path POST `sendChatAction`; every
  request under `@respx.mock` must be routed, so the existing
  `test_text_message_drives_orchestrator_and_replies` must gain a sendChatAction
  stub (lesson: a green respx test breaks when a new call is unmocked).
- `mypy .` type-checks `examples/` too (flake `mkCheck "mypy" "mypy ."`), so the
  example script must be fully typed like `examples/comms_loop.py`.
- e2e realism: the bot's Bot API calls are plain request/response (respx is fine,
  no real socket needed); the orchestrator turn is in-process (supervisor + mock
  backend), so `test-streaming-over-a-real-socket` does not apply here.

## Steps

- [x] `scufris/telegram.py`: add a pure `render_reply(text, tool_calls) -> str`
      that returns `text` unchanged when `tool_calls` is empty, else appends a
      blank line + one ASCII footer line summarizing the calls (unique tool names
      in call order with `xN` counts for repeats, and a `(failed)` marker when any
      call of that tool has a non-`ok`/`success` status). Import `ToolCall` from
      `.sessions` (no import cycle).
- [x] `scufris/telegram.py`: add the typing action. `_send_chat_action(chat_id)`
      POSTs `sendChatAction {chat_id, action: "typing"}`; in `_dispatch`'s
      on_message branch, send one action immediately, spawn a `_keep_typing`
      keepalive (re-send every `_TYPING_INTERVAL` ~4s), `await on_message`, then
      cancel+await the keepalive in a `finally` before `_send(reply)`. Commands
      (`/new`,`/help`) keep replying with no typing action.
- [x] `scufris/app.py`: in `build_telegram_callbacks.on_message`, render the reply
      via `render_reply(done.reply.text, done.reply.tool_calls)` before the
      empty-coalesce (import `render_reply` from `.telegram`).
- [x] `tests/test_telegram.py` (transport): updated
      `test_text_message_drives_orchestrator_and_replies` and
      `test_offset_advances_past_processed_updates` to stub `sendChatAction`; added
      `test_text_turn_shows_typing_action` / `test_commands_send_no_typing_action`;
      added pure-function tests for `render_reply` (no tools -> unchanged; one/many
      tools -> footer with counts; a failed call -> `(failed)`; empty text + tools
      -> footer only).
- [x] `tests/test_telegram.py` (e2e): `test_end_to_end_receive_turn_reply` boots
      the REAL app (mock backend orchestrator, token + allowlist set) with a
      `MockBackend.stream` override emitting `StreamTool` + a reply carrying
      `tool_calls`; runs `app.router.lifespan_context(app)` so `_start_telegram_bot`
      builds the bot with the real `_launch_agent_turn`/`_drain_turn` callbacks;
      drives ONE `poll_once()` (getUpdates + sendMessage + sendChatAction stubbed)
      and asserts the reply carries the mock text + tool footer with a typing
      action observed. Proves receive -> real orchestrator turn -> reply.
- [x] `examples/telegram_bot.py`: a self-contained, human-readable walkthrough
      (docstring + printed steps like `comms_loop.py`) that boots the real app +
      mock backend (tool-emitting stream override), stubs the Bot API with respx,
      drives one text message through the real bot via `poll_once`, and prints the
      typing action + rendered reply (with tool footer). Exit 0 on success. Fully
      typed (mypy `.` covers examples/).
- [x] Full check suite green: `ruff format` (my changed files) + `ruff check .`
      clean; `mypy .` `Success: no issues found in 52 source files`; full
      `python -m pytest` `486 passed`; `python examples/telegram_bot.py` exits 0.

## Changes (as built)

- `scufris/telegram.py`: `render_reply(text, tool_calls)` (pure, ASCII-only tool
  footer), plus the typing action - `_send_chat_action` and a `_keep_typing`
  keepalive spawned around the `on_message` await in `_dispatch`, cancelled in a
  `finally`. One action is sent up front so even a fast turn shows typing (and the
  test is deterministic); the keepalive covers longer turns. Module docstring
  updated to cover rendering.
- `scufris/app.py`: `on_message` renders via `render_reply(done.reply.text,
  done.reply.tool_calls)` before the empty-reply coalesce. `OnMessage -> str` seam
  unchanged, so the bot stays display-agnostic.
- `tests/test_telegram.py`: transport tests for typing + `render_reply`; the e2e
  drives the real turn path via `poll_once` with `TelegramBot.run` stubbed to a
  no-op. Design note: the first cut ran the production free-running `run()` loop
  under the lifespan, but respx serves getUpdates instantly, so the loop
  busy-spun and the process hung (killed at 200s). `poll_once()` is the transport's
  documented test seam and gives a deterministic single receive->turn->reply while
  still exercising the REAL orchestrator callbacks. The example uses the same seam.
- `examples/telegram_bot.py`: new runnable walkthrough (real app + mock backend +
  respx-stubbed Bot API), printing the typing action and the rendered reply.
- No new config fields (T5 adds no settings), so `.env.example` is untouched.

## Definition of Done

1. `render_reply` renders a text-only reply unchanged and appends a compact
   ASCII tool-summary footer when the turn made tool calls (counts for repeats,
   a failed-call marker). (test: `tests/test_telegram.py` render_reply cases)
2. A text turn shows a "typing..." action: the bot sends at least one
   `sendChatAction {action: "typing"}` while `on_message` runs; `/new` and
   `/help` send none. (test: `tests/test_telegram.py` typing)
3. The orchestrator callback renders through `render_reply`, so a real turn's
   tool calls surface in the Telegram reply. (test: e2e assertion on the footer)
4. End-to-end through the REAL app + mock backend: one getUpdates text update
   drives an orchestrator turn and produces a sendMessage whose body carries the
   reply text + tool footer, with a typing action observed.
   (test: `tests/test_telegram.py` e2e)
5. `examples/telegram_bot.py` boots the real app against a stubbed Bot API + mock
   backend and drives receive -> turn -> reply, exiting 0.
   (cmd: `python examples/telegram_bot.py`)
6. Full check suite green (ruff format + check, mypy adds no new errors vs base,
   pytest, and the example script exits 0).
