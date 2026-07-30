# Add host approvals over Telegram

- STATUS: OPEN
- PRIORITY: 43
- TAGS: feature, v0.2.0, host, telegram, agents

## Story

As the operator, I want a pending host action to reach me on Telegram and be
decidable there, so that a machine waiting on me is not a machine waiting until I
next open a laptop.

The credential question is settled: an allowlisted chat IS the operator, and the
audit records which surface decided (`tasks/20260729-125040/DECISION.md` section
3). "No privileged shortcut for being on Telegram" therefore does not mean extra
friction on the phone - it means there is no SECOND decision path: this surface
calls the same `HostApprovalService` as the web routes, so every rule after the
actor is derived is the same code.

## Steps

- [ ] Notify the allowlisted chats when a proposal enters the queue: the risk
      class, what it would run, the preview, who asked, the expiry, and the
      undo / NO-UNDO line - readable on a phone and MarkdownV2-escaped through
      the existing scrub/fallback path.
- [ ] Handle `callback_query` updates. `_handle_update` currently ignores
      anything that is not a text message, so the inline-keyboard path is a new
      update type: dispatch it, honour the same chat allowlist, and answer the
      callback so the client stops spinning.
- [ ] Approve / deny from the keyboard, with the confirmation proportionate to
      the risk: a one-way action needs a second, differently-worded tap that
      carries the acknowledgement token the core requires, and a reversible one
      states its undo in the same message.
- [ ] Attach a denial reason from the chat and let it reach the requesting agent
      (a force-reply or `/deny <id> <reason>`), and add `/approvals` to list the
      pending queue on demand.
- [ ] Go through the ONE approval service with actor
      `operator:telegram:<chat_id>`; a chat outside the allowlist is ignored as
      it is today, and Telegram gains no rule the web path does not have.
- [ ] Handle the races and the edges: a proposal decided on the web edits the
      Telegram message to say who decided it, a stale button tap is refused
      rather than re-run, and expired / drifted / cancelled-run states render as
      themselves.
- [ ] Report the outcome back into the chat, including a failed multi-step
      apply's partial-step detail (for an activation: this boot and the next boot
      disagree).

## Definition of Done

- A host agent proposes, the operator approves from either surface, and the
  action applies exactly once
  (test: `test_host_approval_from_either_surface`).
- Telegram approval enforces the same gate as the web path - one service, one set
  of refusals, the actor being the only difference
  (test: `test_telegram_approval_uses_the_same_enforcement`).
- A one-way action needs the second tap; the first tap alone never applies it
  (test: `test_telegram_one_way_needs_the_second_tap`).
- A button tap on an already-decided or expired proposal is refused and says why,
  and nothing runs twice
  (test: `test_telegram_stale_button_is_refused`).
- A denial from Telegram reaches the requesting agent with its reason
  (test: `test_telegram_denial_reaches_the_requesting_agent`).
- A chat outside the allowlist cannot decide anything
  (test: `test_telegram_disallowed_chat_cannot_approve`).
- cmd: `python -m pytest`
- cmd: `nix flake check`
- manual: approving a real host change from a phone is clear enough to do
  confidently while away from the desk.

## Notes

- Epic: 20260729-124655. Depends on 20260729-125040 (the approval service) and is
  the task that closes the epic's "either surface" criterion.
- Re-cut from 20260729-125040 - see `tasks/20260729-125040/DECISION.md` section 1.
- `scufris/telegram.py` is long-poll `getUpdates` with text commands only; the
  bot is constructed in `app.py` with injected callbacks, so it reaches the
  approval service in-process rather than over HTTP (which is what lets it hold
  an operator identity at all - the machine bearer token is refused on the
  decision routes by design).
- The bot already streams orchestrator turns per phase; an approval should read as
  part of that conversation, not as a second bot.

## Flow State

- FLOW STEP: PLANNING
- PLAN STATUS: APPROVED
