# T6: Telegram live turn streaming - thinking bubble + per-tool widgets + phased answer

- STATUS: CLOSED
- PRIORITY: 34
- TAGS: telegram, feature, ui, streaming, agent
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Goal

As a Telegram user, when I message the bot I want to see the orchestrator's turn
unfold live instead of waiting for one silent reply: a "thinking" bubble that
streams the orchestrator's full reasoning, a discrete widget message per tool
call as it completes, then the final answer as its own message. This is the
edited-message live-streaming polish the Telegram spike deferred
(tasks/20260722-221359/SPIKE.md, Q5 + open question "Reply streaming fidelity").

Observable done: sending one message produces, in the chat, (1) a Thinking
message that gets edited as reasoning streams, (2) one tool message per tool the
turn ran, (3) a final answer message - in that chronological order.

## Notes

- Spike: tasks/20260722-221359/SPIKE.md (Q5 rendering; the deferred streaming
  polish). Builds directly on T5 (tasks/20260722-222739) which shipped the
  final-per-turn render + typing action + example + e2e.
- DECISION.md (this folder) records the user-approved rendering shape:
  message-per-phase layout + full streamed reasoning + emoji widgets.
- Harness-first (AGENTS.md): respx-stubbed Telegram + mock backend; extend the
  `examples/telegram_bot.py` walkthrough to stream reasoning + tools.

## Understanding (grounded in the code, 2026-07-26)

The streaming events already exist and already flow through the run's EventBus;
only the Telegram bot fails to render them (it drains them and shows just the
final text). So this is additive wiring on a verified seam, not new streaming
machinery.

What exists:
- `scufris/agent.py` streams per-turn `StreamEvent`s: `StreamReasoningDelta`
  (the "thinking" tokens, from codex `item/reasoning/textDelta`), `StreamTextDelta`
  (answer tokens), `StreamTool` (a tool completed: `ToolCall(server, tool, status)`),
  `StreamSessionStarted`, `StreamDone` (final `AgentReply`), `StreamError`. Confirmed
  by lesson `codex-app-server-for-token-streaming`.
- `scufris/eventbus.py` `EventBus.subscribe(after_seq=0)` yields `(seq, event)` live;
  the web SSE relay already consumes this. The Telegram bot does NOT - `app.py`
  `build_telegram_callbacks.on_message` calls `_drain_turn(bus)`, which throws away
  every intermediate event and returns only `StreamDone`, then renders one message.
- `scufris/telegram.py` `TelegramBot`: long-poll transport, allowlist, `/new`|`/help`
  dispatch, a "typing..." keepalive around the turn, `render_reply(text, tool_calls)`
  (ASCII tool-summary footer), one `sendMessage` per turn. Callback seam:
  `OnMessage = Callable[[str], Awaitable[str]]`.
- `_launch_agent_turn` returns `(run_id, bus)`; the orchestrator turn runs as a
  supervised background job publishing events on `bus`.

The change (message-per-phase live render; see DECISION.md):
1. **Callback seam becomes streaming.** Replace `OnMessage -> str` with
   `OnMessageStream = Callable[[str], AsyncIterator[StreamEvent]]`. `app.py`'s
   `build_telegram_callbacks.on_message` becomes an async generator: `launch_turn`,
   then `async for (_seq, event) in bus.subscribe(): yield event` until `StreamDone`.
   ALL app-level error/edge cases are mapped to a friendly terminal
   `StreamError(detail=...)` inside the generator (agent disabled; 409 busy ->
   "still working on the previous message..."; launch failure and a BACKEND
   `StreamError` -> "that turn failed") so the raw technical detail never reaches
   the chat and the bot stays display-agnostic (renders `StreamError.detail`
   verbatim as a plain message).
2. **The bot owns rendering** (`telegram.py`), keeping all Bot API calls in one
   unit-testable place. New `_render_turn(chat_id, events)` consumes the async
   iterator and drives Telegram per phase:
   - `StreamReasoningDelta`: append to the current reasoning buffer. If no reasoning
     bubble is open, `sendMessage` a new Thinking message and remember its
     `message_id` (first paint immediate - lesson
     `dont-gate-streaming-render-on-a-single-raf`). Otherwise `editMessageText`
     the bubble, THROTTLED to >= `edit_interval` since the last edit and only when
     the text changed (Telegram 429s on rapid edits and errors on an unmodified
     edit). Reasoning is HTML-escaped, italicised, and windowed to the tail
     (Telegram's 4096-char cap).
   - `StreamTool`: flush+finalize the open reasoning bubble (final edit), CLOSE it
     (so the next reasoning delta opens a NEW bubble below), then `sendMessage` a
     discrete tool widget: emoji + tool name + a check/cross by status. (StreamTool
     carries only name+status; per-tool RESULT text is out of scope for v1 - the
     shared StreamEvent/ToolCall does not carry it. Noted honestly, not faked.)
   - `StreamDone`: finalize the open reasoning bubble; `sendMessage` the final
     answer as its own PLAIN-text message via `render_reply(reply.text,
     reply.tool_calls)` (plain, no parse_mode - the model's free text may contain
     `<`/markdown that HTML parse_mode would reject; unchanged from T5). Empty ->
     "(the orchestrator returned no text)".
   - `StreamError`: `sendMessage` `detail` as a plain message; stop.
   The interleave contract (thinking#1 -> tool A -> thinking#2 -> tool B -> answer)
   IS the message-per-phase layout: a tool always closes the current reasoning
   bubble so chat order stays chronological.
3. **Widgets use emoji + HTML** for the Thinking + tool messages only (a
   deliberate, DECISION.md-recorded exception to the repo's ASCII-only convention,
   scoped to the Telegram rendered surface; code/comments/commits/docs stay ASCII).
   The final answer message stays plain ASCII (render_reply) as today.
4. **Config toggle** `telegram_stream` (default True): when False, `_render_turn`
   ignores reasoning/tool events and renders only the final answer (the T5
   behaviour), a safety valve the spike wanted for the post-hands-on UX call.
   No second code path - just a gate inside `_render_turn`.

Seam / verification notes (grounded in code + lessons):
- Keeps the typing action (T5) around the whole turn.
- `_drain_turn` stays - the landing chat and fork still use it; only Telegram
  switches to `bus.subscribe`.
- Every new Bot API call (`editMessageText`) must be respx-routed in EVERY test
  that drives a turn, or a previously-green test breaks (T5 lesson: a new unmocked
  call fails an @respx.mock test). Make the bot's `edit_interval` a constructor
  param so tests set it to 0 and drive deterministic edits.
- `mypy .` covers `examples/`, so the example stays fully typed.
- Existing `_Recorder.on_message` and `build_telegram_callbacks` tests change with
  the streaming seam (fakes yield StreamEvents).

## Steps

- [x] `scufris/telegram.py`: change the callback seam to
      `OnMessageStream = Callable[[str], AsyncIterator[StreamEvent]]`; keep
      `OnReset`. Add pure formatters: `_format_reasoning(buf) -> str` (HTML-escape,
      italic, tail-window to fit 4096 with a Thinking header) and
      `_format_tool(call) -> str` (emoji + tool name + status check/cross). Keep
      `render_reply` for the final answer.
- [x] `scufris/telegram.py`: add `_send_message(chat_id, text, *, html=False) -> int`
      (returns the sent `message_id`) and `_edit_message(chat_id, message_id, text,
      *, html)`; add an `edit_interval` constructor param (default ~1.5s). Implement
      `_render_turn(chat_id, events)` per the phase contract above (open/edit/close
      reasoning bubble, tool widget message, final answer, StreamError). Wire it into
      `_dispatch`'s on_message branch inside the existing typing keepalive
      try/finally. Add a `stream` flag (default True) gating reasoning/tool rendering.
      Update the module docstring.
- [x] `scufris/config.py`: add `telegram_stream: bool = True`
      (`SCUFRIS_TELEGRAM_STREAM`). `.env.example`: document it.
- [x] `scufris/app.py`: rewrite `build_telegram_callbacks` to build the streaming
      `on_message` async generator over `launch_turn` + a `stream_turn`/`bus.subscribe`
      helper (replacing the `_drain_turn` call), mapping disabled/409/failure/backend-
      StreamError to a friendly terminal `StreamError`. Pass `settings.telegram_stream`
      + edit interval into `TelegramBot` in `_start_telegram_bot`. `_drain_turn` stays
      for the landing chat/fork.
- [x] `tests/test_telegram.py`: update the `_Recorder`/`_make_bot`/
      `build_telegram_callbacks` fakes to the streaming seam (async-generator
      on_message); route `editMessageText` in every turn-driving test. Add pure tests
      for `_format_reasoning` (escape, italic, tail-window) and `_format_tool`
      (ok/failed). Add render tests with a fake event stream (`edit_interval=0`):
      reasoning -> a Thinking sendMessage + editMessageText(s); a `StreamTool` ->
      closes the bubble + a tool sendMessage; `StreamDone` -> a final answer
      sendMessage; assert message-per-phase ORDER. Add a `StreamError` render test
      (detail sent as a plain message). Add a `stream=False` test (only the final
      answer is sent).
- [x] `tests/test_telegram.py` (e2e): extend `test_end_to_end_receive_turn_reply`
      (or add a streaming sibling) so the `MockBackend.stream` override emits
      `StreamReasoningDelta` + `StreamTool` + `StreamDone`; run the real `_lifespan`
      + `poll_once`; assert a Thinking message, a tool message, and a final answer
      message were sent through the real `_launch_agent_turn` + `bus.subscribe`
      callbacks.
- [x] `examples/telegram_bot.py`: stream reasoning + a tool + done from the mock
      backend override and print the phased messages (Thinking edits, tool widget,
      answer). Exit 0. Fully typed.
- [x] `CHANGELOG.md`: add an entry under the current unreleased section.
- [x] Full check suite green: `ruff format` (changed files) + `ruff check .`;
      `mypy .` adds no new errors vs base; full `python -m pytest`;
      `python examples/telegram_bot.py` exits 0.

## Changes (as built)

- `scufris/telegram.py`: callback seam is now
  `OnMessageStream = Callable[[str], AsyncIterator[StreamEvent]]`. New pure
  formatters `_format_reasoning` (HTML-escaped, italic, tail-windowed under 4096
  with a brain-emoji Thinking header) and `_format_tool` (wrench + tool name +
  a heavy-check/cross by status; shows `server.tool` for a non-scufris server).
  `_send_message(...) -> int | None` returns the sent `message_id`; `_edit_message`
  edits it best-effort (swallows a 429/"unmodified" 400). `_render_turn` drives the
  message-per-phase render: open a live Thinking bubble on the first reasoning
  delta (first paint immediate, then throttled `_edit_message`s), a `StreamTool`
  force-flushes and CLOSES the bubble then sends a tool widget, `StreamDone` sends
  the final answer via `render_reply` (plain), `StreamError` sends its `detail`.
  `stream` + `edit_interval` constructor params gate/throttle it. Emoji are
  `\N{...}` escapes so the SOURCE stays ASCII. Module docstring rewritten.
- `scufris/config.py` + `.env.example`: `telegram_stream: bool = True`
  (`SCUFRIS_TELEGRAM_STREAM`).
- `scufris/app.py`: `build_telegram_callbacks` dropped its `drain_turn` param and
  `on_message` is now an async generator - it `launch_turn`s then forwards
  `bus.subscribe(after_seq=0)` events until `StreamDone`, mapping agent-disabled /
  409-busy / launch-failure / a backend `StreamError` to a friendly terminal
  `StreamError` (raw detail never leaks to the chat). `_start_telegram_bot` passes
  `stream=settings.telegram_stream`. `_drain_turn` is unchanged (landing chat/fork).
- `tests/test_telegram.py`: fakes moved to the streaming seam; new render tests
  (phase order; a second tool opening a fresh bubble; `stream=False` -> answer
  only; `StreamError` -> plain detail; empty answer coalesced), formatter unit
  tests, and callback tests over a `_FakeBus`. The e2e
  (`test_end_to_end_receive_stream_reply`) streams reasoning + tool + done through
  the real app and asserts the Thinking/tool/answer messages.
- `examples/telegram_bot.py`: the mock stream now reasons + calls a tool; the
  script prints the phased render (verified: Thinking bubble, tool widget, answer).
- `CHANGELOG.md`: Added entry.

Verification (in the worktree, via `nix develop`): `ruff check .` clean; `mypy .`
`Success: no issues found in 52 source files`; full `python -m pytest` all passed;
`python examples/telegram_bot.py` exit 0; touched files ASCII-clean
(`grep -nP "[^\x00-\x7f]"` -> none; emoji are `\N{}` escapes).

## Definition of Done

1. A text turn renders message-per-phase: a Thinking message is sent and then
   edited as reasoning streams; each `StreamTool` produces a discrete tool widget
   message; `StreamDone` produces the final answer message - in chronological
   order. (test: `tests/test_telegram.py` streaming-render cases)
2. Reasoning rendering is safe: HTML-escaped, tail-windowed under 4096 chars, and
   edits are throttled to the configured interval and skipped when unchanged.
   (test: `_format_reasoning` unit cases + an edit-throttle/ordering render test)
3. A tool widget shows the tool name and success/failure; the final answer keeps
   the T5 `render_reply` footer and stays plain ASCII text. (test: `_format_tool`
   + final-answer assertion)
4. `telegram_stream=False` falls back to a single final-answer message (no
   reasoning/tool messages). (test: `stream=False` render case)
5. App-level conditions surface as a friendly single message via a terminal
   `StreamError` (agent disabled; 409 "still working..."; turn failed), never a
   raw backend detail. (test: `StreamError` render + a 409/disabled generator test)
6. End-to-end through the REAL app + mock backend: one getUpdates text update
   drives a streamed turn producing Thinking + tool + answer messages.
   (test: `tests/test_telegram.py` e2e)
7. `examples/telegram_bot.py` boots the real app against a stubbed Bot API + mock
   backend and streams a phased turn, exiting 0.
   (cmd: `python examples/telegram_bot.py`)
8. Full check suite green (ruff format + check, mypy adds no new errors vs base,
   pytest, and the example exits 0).
