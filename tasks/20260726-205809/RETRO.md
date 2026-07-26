# Retro: Render bot markdown for Telegram (markdown -> MarkdownV2 reply)

- TASK: 20260726-205809
- BRANCH: feature/telegram-markdown-reply
- REVIEW ROUNDS: 1 (APPROVE, out-of-context)

Process notes only; what/why/evidence live in TASK.md close-out, findings in
REVIEW.md.

## What went well

- Verified the library's ACTUAL output before writing any assertion: ran
  `telegramify_markdown.markdownify` on a table/list/heading sample in the dev
  shell and designed the tests against the real MarkdownV2 (bullet glyph `⦁`,
  `host\_stats` escaping, fenced table) instead of guessing. No churn from
  wrong-shape assertions.
- The two-layer "never drop a reply" safety (converter fallback + transport
  fallback) was specified at plan time as load-bearing and each layer got its
  own independently-failing test; the out-of-context reviewer re-derived both
  and found nothing to add.
- Doc-surface sweep ran and paid off immediately: the CHANGELOG's "final answer
  stays plain text" claim was now false and was caught and fixed in the same
  task, not left to rot.
- Clean single review round: an out-of-context reviewer with all five DoD proofs
  executed returned APPROVE with only two optional NITs.

## What went wrong

- The plan's DoD proof filters did not match the tests as written: `-k
  "fallback"` selected ZERO tests (the tests are named `..._falls_back_to_plain`,
  no substring "fallback"), and `-k markdown_reply` missed the parse-mode send
  test. Root cause: the `-k` filters were written into the DoD at plan time,
  before the tests existed, so they were guesses about future test names. Caught
  only because I ran each DoD grep explicitly before closing (a proof that
  selects nothing proves nothing) - had I trusted the plan text, DoD #2 would
  have "passed" against an empty selection.

## What to improve next time

- When a DoD proof is a `-k`/grep filter, treat "the filter selects >0 of the
  intended tests" as part of the proof: either name the tests to match the
  planned filter up front, or finalize the filter AFTER the tests exist and
  confirm the selection count. Never close on a `-k` proof without seeing it
  select the specific tests it claims to verify.

## Action items

- [x] Ledger: added `dod-kfilter-proof-must-select-tests` (sibling of
  `scope-absence-greps-to-the-diff-not-the-file`).
- No follow-up code tasks. The two NITs (broad `except` in `_send_reply`,
  `warning` vs DEBUG log level) were intentional and responded to in REVIEW.md;
  nothing to file.
