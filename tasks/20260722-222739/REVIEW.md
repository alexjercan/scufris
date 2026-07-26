# Review: T5 Telegram reply rendering + end-to-end example

- TASK: 20260722-222739
- BRANCH: feat/telegram-t5-rendering

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (findings), confirmed in-session

Round-1 findings were produced by a fresh reviewer with no sight of the
implementing session. It read the T5 TASK.md, the spike Q5, and the T4 seam
contract, reviewed the full `master...feat/telegram-t5-rendering` diff, read all
four changed files, and ran the suite in the nix devshell: `ruff check .` clean,
`mypy .` "Success: no issues found in 52 source files", `python -m pytest` 486
passed, and `python examples/telegram_bot.py` printed the typing action + the
rendered reply (text + `tools: host_stats` footer) and exited 0. It judged the
diff to deliver the Goal, the `render_reply` tests meaningful (exact-string
assertions that break if the feature is removed), and the e2e a faithful real-turn
exercise (only the poll-loop scheduling is stubbed; `_launch_agent_turn`/
`_drain_turn`/supervisor/MockBackend all run) with the `run`-stub honestly
documented. In-session I re-derived the one load-bearing claim behind R1.1: in
`poll_once`, the offset advances (telegram.py:168) BEFORE `_handle_update` ->
`_dispatch`, so an exception from the upfront typing send does drop the update.
Confirmed. Verdict APPROVE (no BLOCKER/MAJOR); R1.1 addressed anyway as a cheap
robustness win.

- [x] R1.1 (MINOR) scufris/telegram.py:218 - The upfront `_send_chat_action`
  raised on HTTP error and was not failure-tolerant like the keepalive, so a
  transient sendChatAction failure aborted `_dispatch` after the offset had
  already advanced, silently costing the user their reply.
  - Response: Fixed. Factored `_try_typing(chat_id)` that swallows
    non-cancellation errors (logs at DEBUG) and used it for BOTH the upfront send
    and the keepalive; `_send_chat_action` still raises so `_try_typing` owns the
    tolerance. Pinned by `test_typing_action_failure_does_not_block_reply` (a 500
    from sendChatAction, reply still sent). Full suite re-run green.

No `manual:` DoD items on this task, so there are no pending user acceptance
checks.
