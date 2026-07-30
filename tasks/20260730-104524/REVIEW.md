# Review: Add host approvals over Telegram

- TASK: 20260730-104524
- BRANCH: feat/telegram-approvals

## Round 1

- VERDICT: REQUEST_CHANGES
- REVIEWER: in-session (no out-of-context mechanism available - subagents are
  disabled in this session, so the round-1 default could not be used. Recorded as
  the exception the review skill allows. The MAJOR below came from PROBING the
  renderer with a realistic long preview, and the probe output is quoted.)

What was verified rather than taken on trust:

- `python -m pytest` 854 passed; `nix flake check` all four checks. The 13 new tests
  drive a REAL hostd on a real socket and the REAL bot the app started (not a second
  one built beside it), so a tap travels to the helper's executor.
- Both `examples/telegram_approval.py` flows were run and READ: the one-tap
  reversible case and the two-tap one-way case, including the armed message and the
  edited record afterwards.
- The audit was checked to name the surface: `operator:telegram:4242` on the
  approved and applied lines.
- The allowlist refusal was checked at BOTH layers (the bot drops the tap; the
  providers refuse the same chat id when called directly).

- [x] R1.1 (MAJOR) scufris/telegram.py `render_approval` - the message cap trims the
  TAIL, which is where the undo line lives, so for a long preview the operator is
  shown the diff and NOT how it can be undone. Probed with a 60-line preview (a
  closure diff is routinely longer): the rendered body comes back at exactly 4096
  characters ending
  `xxxxxxxxx\n\n[trimmed - read the whole thing on the dashboard's /host/ page]` -
  the `NO UNDO: ...` sentence and any `RESULT:` line are gone. This bites hardest on
  the highest-risk class: an R3 activation's preview IS a closure diff, so the action
  whose undo sentence matters most is the one most likely to lose it. Suggested
  change: trim the PREVIEW BODY rather than the message tail - re-render from a copy
  of the record whose `preview.lines` are shortened (with the omission stated), so
  the head, the commands, the undo line and the result always survive. Pin it with a
  test that a long preview still contains the undo sentence.
  - Response: fixed. `render_approval` now shortens the preview lines on a deep COPY
    of the record and re-renders (so there is still ONE renderer), stating how many
    lines were elided; the head, the commands, the undo line and the result survive by
    construction. A tail cut remains only as the last resort for a body that does not
    fit even with no preview at all, which is better than a message Telegram refuses.
    Pinned by `test_a_long_preview_keeps_the_undo_line`, driven with the ~12k-character
    preview the probe used.
- [x] R1.2 (MINOR) scufris/telegram.py `TelegramBot._announced` /
  `_reason_prompts` - both grow for the process lifetime. `_announced` gains an entry
  per announced action (and per `/approvals` listing), and `_reason_prompts` keeps an
  entry for every Deny tap whose prompt is never answered - probed: after a Deny tap
  the map holds one entry, and denying the same action on the web leaves it there.
  The consequence today is small (a bounded number of small tuples on a bot that
  restarts with the app, and answering a stale prompt is refused honestly by the
  service), but a decision surface on a long-lived process should not accumulate
  state without a ceiling. Suggested change: cap both (drop the oldest, as
  `HostActionStore` does) or clear an action's entries when it is decided.
  - Response: both are `OrderedDict`s capped at `MAX_TRACKED_ACTIONS` (200, matching
    the app's own registry), oldest first, through the two small helpers that own the
    writes (`_remember` / `_await_reason`). Pinned by
    `test_the_bot_does_not_accumulate_tracked_actions`.
- [x] R1.3 (NIT) scufris/telegram.py `_handle_update` - a reply to the deny prompt is
  taken as the reason even when it starts with `/`, so an operator who replies
  `/cancel` denies the action with the reason "/cancel" instead of cancelling
  anything. They were asked for a reason, so this is defensible; a one-line guard
  (treat a leading `/` as a command and re-ask) would be kinder.
  - Response: done - a reply that parses as a command is answered with
    `REASON_STILL_WANTED` and the prompt stays OPEN, so the reason can still be given
    (pinned by `test_a_command_replied_to_the_prompt_is_not_a_reason`, which also
    asserts the real reason lands afterwards).

Pending user checks (not resolved by this review):

- manual: approving a real host change from a phone is clear enough to do
  confidently while away from the desk. This session has no Telegram account or
  phone, so `examples/telegram_approval.py` is the substitute: it prints every
  message and button in order for both flows, and reading that is the closest this
  review gets to the real thing.

## Round 2

- VERDICT: APPROVE
- REVIEWER: in-session (same exception as round 1; each fix re-verified by running
  its pin, and R1.1 re-derived by re-running its probe as an assertion.)

All three findings are resolved and ticked. R1.1's pin fails if the tail-trim is
restored, because the undo line is what it asserts. Gate after the fixes:
`python -m pytest` 857 passed, `nix flake check` all four checks,
`examples/telegram_approval.py` re-run in both flows.

Pending user checks (not resolved by this review):

- manual: approving a real host change from a phone is clear enough to do
  confidently while away from the desk. No Telegram account or phone in this
  session; `examples/telegram_approval.py` prints every message and button for both
  flows and is the substitute.
