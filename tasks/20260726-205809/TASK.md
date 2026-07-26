# Render bot markdown for Telegram (tables/lists/headings) via a markdown->MarkdownV2 wrapper on the reply

- STATUS: OPEN
- PRIORITY: 30
- TAGS: telegram,feature,ui,rendering,markdown

## Flow State

- FLOW STEP: PLANNING
- PLAN STATUS: PENDING

## Goal

As a Telegram user, when the bot sends its final answer I want markdown in that
answer (tables, lists, headings, bold/italic, code, links, blockquotes) to be
displayed in a formatted way that Telegram renders properly, instead of the raw
markdown text I see today.

The formatting is applied ON the bot response - a `markdown -> Telegram`
wrapper over the model's text - not by instructing the model to change what it
emits. The model keeps producing normal GitHub-flavored markdown; the bot
transforms it at render time.

Observable done: sending a prompt whose answer contains a markdown table, a
bulleted/numbered list, and a heading produces a Telegram message where the
table is shown as an aligned monospace block, the list shows as bullets/numbers,
and the heading shows as bold - not as literal `|`/`#`/`-` characters. A reply
whose markdown is malformed still arrives (as plain text), never dropped.

## Decision (user-approved 2026-07-26)

Converter = the `telegramify-markdown` PyPI library, output = Telegram
**MarkdownV2**. Chosen over an owned mistune->HTML renderer and a hand-rolled
regex converter. It is purpose-built for exactly this (GFM -> MarkdownV2:
headings->bold, lists->bullets, tables->fenced code blocks, full MarkdownV2
escaping handled), so we own less fragile parsing code. See the flow gate
question/answer; recorded here as the load-bearing build-shape choice.

Trade-off accepted: the final answer is sent with `parse_mode=MarkdownV2`, a
different parse mode from the existing HTML thinking/tool widgets. That is fine -
each Telegram message sets its own parse mode independently; the widgets are
unchanged.

## Understanding (grounded in the code, 2026-07-26)

Today the FINAL answer is the only un-formatted surface. `scufris/telegram.py`:

- `_render_turn` (the `StreamDone` branch, ~line 386-390) builds the answer with
  `render_reply(event.reply.text, event.reply.tool_calls)` and sends it via
  `self._send_message(chat_id, body)` with **no `parse_mode`** - deliberately
  plain text (module docstring lines 22-24, and `render_reply`'s docstring): the
  model's free text may contain `<`/markdown that an HTML parse mode would 400 on,
  which would DROP the whole reply. So the current safety story is "plain text
  never 400s on formatting".
- `render_reply(text, tool_calls)` returns the model text plus, when tools ran, a
  blank line and an ASCII `tools: a, b (failed)` footer. ASCII only.
- The thinking bubble (`_format_reasoning`) and tool widgets (`_format_tool`)
  already use `parse_mode=HTML` with manual `html.escape`; they are NOT in scope.
- `_send_message(chat_id, text, *, html=False)` posts `sendMessage`, sets
  `parse_mode=HTML` when `html=True`, and `raise_for_status()`es (raises on 400).

Constraint that shapes everything: Telegram's Bot API has **no tables, no
headings, no native lists**. Its vocabulary is bold/italic/underline/strike/
spoiler, inline code, `pre` code blocks, links, blockquotes. So the wrapper must
TRANSFORM markdown, and a table can only be shown as an aligned monospace code
block. `telegramify-markdown` does all of this.

Robustness is load-bearing: replacing "plain text never 400s" with "send
MarkdownV2" reintroduces the 400 risk. So the wrapper MUST keep the guarantee
that a reply is never dropped:
1. converting text: wrap `markdownify()` in try/except; on ANY exception fall
   back to the raw plain body;
2. sending: try `parse_mode=MarkdownV2`; on a Telegram 4xx fall back to sending
   the plain body with no parse mode.
The existing plain `render_reply(...)` output is exactly that fallback body.

Note (explicitly OUT of scope, pre-existing): messages over Telegram's 4096-char
cap already 400 today and are not split; this task does not add message
splitting. `telegramify.telegramify()` (the chunking API) is a possible future
task. We use only `telegramify_markdown.markdownify()`.

## Plan

Add the dependency, add a `markdown_reply` render function next to `render_reply`
(so the plain output stays the fallback), and make the `StreamDone` send try
MarkdownV2 then fall back to plain. Only the final-answer path changes.

## Steps

- [ ] `nix develop` then `uv add telegramify-markdown`; `uv lock`; re-enter the
      dev shell so uv2nix picks up the new wheel. Confirm `import
      telegramify_markdown` works and `nix flake check` still builds the venv
      (transitive deps `mistletoe`, `emoji` are pure-Python wheels; if any needs
      a build-system override, add it to the flake overlay).
- [ ] In `scufris/telegram.py` add `markdown_reply(text, tool_calls) -> str`:
      builds the same combined body as `render_reply` (model text + optional
      `tools:` footer) but returns it converted to MarkdownV2 via
      `telegramify_markdown.markdownify(...)`. Wrap the conversion in try/except
      and, on failure, return `render_reply(...)` unchanged (plain) so a converter
      bug can never lose the reply. Document the escape/transform contract in the
      docstring (headings->bold, lists->bullets, tables->code block).
- [ ] Add a send helper `_send_reply(chat_id, markdown_body, plain_body)`: POST
      `sendMessage` with `parse_mode=MarkdownV2`; on a non-2xx response re-POST
      `plain_body` with no parse mode (and log the fallback at DEBUG). Reuse it
      only for the final answer.
- [ ] Rewire the `StreamDone` branch of `_render_turn` to compute both
      `plain = render_reply(...)` and `md = markdown_reply(...)` and send via
      `_send_reply(chat_id, md or EMPTY_REPLY, plain or EMPTY_REPLY)`. Leave the
      `StreamError` branch on plain `_send_message` (its `detail` is a friendly
      ASCII line, no markdown).
- [ ] Tests in `tests/test_telegram.py`:
      - unit: `markdown_reply` on a body with a heading, a bulleted list, a
        numbered list, a table, bold/inline-code and a link -> assert the
        MarkdownV2 output renders the table as a fenced/monospace block, the
        heading as bold, the list as bullets, and that MarkdownV2 specials are
        escaped (no bare `.`/`-`/`|` that would 400).
      - unit: `markdown_reply` preserves the `tools:` footer.
      - fallback (converter): monkeypatch `markdownify` to raise -> assert
        `markdown_reply` returns the plain `render_reply` body.
      - fallback (transport): respx stubs the MarkdownV2 `sendMessage` to 400 and
        the plain retry to 200 -> drive one `StreamDone` and assert the bot
        re-sends the plain body with NO `parse_mode`, and the user still gets it.
      - update the existing final-answer send assertions / e2e for the new
        MarkdownV2 parse mode on the answer message.
- [ ] Extend `examples/telegram_bot.py` so the mock turn's final answer contains
      a table + list + heading, demonstrating the formatted render end to end.
- [ ] Full gate: `ruff check .`, `mypy .`, `pytest`, then `nix flake check`.

## Definition of Done

1. The bot's final-answer message is sent with `parse_mode=MarkdownV2` and its
   markdown is transformed for Telegram: table -> aligned monospace block,
   list -> bullets/numbers, heading -> bold. (test: `pytest
   tests/test_telegram.py -k markdown_reply`)
2. A reply is NEVER dropped by formatting: a converter exception falls back to
   plain text, and a Telegram MarkdownV2 400 falls back to a plain-text resend.
   (test: `pytest tests/test_telegram.py -k "fallback"`)
3. The thinking bubble and tool widgets are unchanged (still HTML). (test:
   `pytest tests/test_telegram.py -k "reasoning or tool"`)
4. `examples/telegram_bot.py` shows a formatted table/list/heading answer end to
   end. (manual: run the example, read the rendered reply block)
5. Full QA gate green. (cmd: `nix flake check`)

## Notes

- Builds on T6 (tasks/20260726-201901) live streaming and T5
  (tasks/20260722-222739) reply rendering + example.
- `telegramify-markdown` API used: `telegramify_markdown.markdownify(content)`
  returns a MarkdownV2 string. Verify the exact signature/kwargs in the dev shell
  at work time (`normalize_whitespace`, `max_line_length` may be relevant for the
  table block width).
