# Review: Render bot markdown for Telegram (markdown -> MarkdownV2 reply)

- TASK: 20260726-205809
- BRANCH: feature/telegram-markdown-reply

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Out-of-context reviewer (fresh subagent, no sight of the implementing session)
ran all five DoD proofs in the worktree and independently re-derived the two
fallback claims. In-session pass re-verified the transport fallback claim itself
(`pytest -k "falls_back"` -> 2 passed) before adopting the round.

DoD proofs run (all PASS):
- DoD 1 `pytest -k "markdown_reply or markdownv2"` -> 7 passed.
- DoD 2 `pytest -k "falls_back"` -> 2 passed.
- DoD 3 `pytest -k "reasoning or tool"` -> 17 passed.
- DoD 4 ran `examples/telegram_bot.py` -> final answer tagged `[send MarkdownV2]`,
  heading bold, list as `⦁` bullets, table as an aligned code block, footer
  `host\_stats`.
- DoD 5 `ruff check .` + `mypy .` clean (52 files); full suite 509 passed. (Full
  `nix flake check` confirmed separately by the implementer; dev-shell build +
  lint/type/tests green.)

Load-bearing claims re-derived: converter exception -> plain body
(`'hello world\n\ntools: host_stats'`); transport 400 -> single plain resend with
no `parse_mode` and no double-delivery. Both confirmed.

Design / honesty / docs / tests: the `StreamDone` empty-answer branch diverges
from plan step 4's literal wording (`if plain:` + plain `EMPTY_REPLY` instead of
`md or EMPTY_REPLY`) - a correct improvement, since `EMPTY_REPLY`'s parens are
MarkdownV2 specials and must not be sent as MarkdownV2; the close-out notes
describe this accurately. Doc sweep confirmed (no stale "plain text" claims
remain). New tests assert real behavior and would fail with the fix removed; the
four pre-existing final-answer assertions were strengthened to the MarkdownV2
form, not weakened.

- [ ] R1.1 (NIT) scufris/telegram.py `_send_reply` - the fallback catch is a
  broad `except Exception`, so a transient network error (e.g.
  `httpx.ConnectError`), not just a Telegram parse rejection, triggers a
  plain-text resend. Defensible (plain is more likely to succeed; the reply is
  not dropped) and `asyncio.CancelledError` is re-raised first. Optional: narrow
  to `httpx.HTTPStatusError` if network errors should propagate as before.
  - Response: Left as-is. The broad catch is intentional: the goal is "a reply
    is never dropped by formatting", and a plain resend is the right response to
    any send failure of the formatted body, not only a parse rejection.
    `CancelledError` is already re-raised, so shutdown is unaffected.
- [ ] R1.2 (NIT) scufris/telegram.py `markdown_reply` - logs the converter
  failure at `warning` while the transport fallback logs at DEBUG. Asymmetry is
  intentional (a converter bug deserves more visibility than a per-message parse
  rejection); noted for the record.
  - Response: Intended. A converter exception is a code-level bug worth
    surfacing; a MarkdownV2 parse rejection is expected-ish per-message noise.

No open manual DoD items beyond DoD 4, which the reviewer executed and confirmed.
