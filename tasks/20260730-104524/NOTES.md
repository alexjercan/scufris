# Notes: host approvals over Telegram

What shipped and why. The credential fork this task builds on is
`tasks/20260729-125040/DECISION.md` section 3 (an allowlisted chat IS the operator).

## What shipped

`scufris/telegram.py` gains the approval surface: `ApprovalOps` / `ApprovalOutcome`
(the injected providers), `render_approval` + `approval_keyboard` +
`confirm_keyboard`, `callback_query` dispatch, the force-reply denial-reason flow,
`/approvals` and `/deny <id> <reason>`, and `announce_proposal` /
`announce_decision`. `scufris/app.py` builds the providers and wires the two
announcement hooks. `examples/telegram_approval.py` prints the whole thing.

Four decisions worth naming:

- **The transport supplies WHO, and nothing else.** The bot hands the app a chat id;
  the app derives `operator:telegram:<chat_id>`, re-checks the allowlist and calls
  the one `HostApprovalService`. So the acknowledgement rule, the expiry, the drift
  check and the cross-surface race have a single implementation, and every refusal
  the operator reads is the service's own sentence rather than something the
  transport invented. The allowlist is checked at BOTH layers, so neither is the only
  thing between a stray chat and a root command.
- **One renderer.** The chat message is `host_actions.render_action` - the same text
  the dashboard and the proposing agent see (`share-one-renderer-so-two-surfaces-cannot-drift`).
  What Telegram adds is the keyboard, not a paraphrase.
- **The one-way case is two taps.** The first tap only ARMS it: the message says what
  cannot be undone and offers a differently-worded confirm. The token still comes
  from the RECORD, never from the callback payload (which is capped at 64 bytes
  anyway), so a tapped button cannot assert its own terms.
- **`decidable()` is now one definition.** A surface offers a control only where a
  decision can still be made; a queue that hands back a button the service would
  refuse is a queue that lies about what the operator can do.

## Difficulties

- **The test harness had to run the bot on the APP's loop.** Awaiting
  `bot._handle_update` from the test's own loop makes `supervisor.start`'s
  `create_task` land on a loop nobody is running, so the approved apply never
  progressed and `_settle` timed out. Driving the update through the client's portal
  (`client.portal.call`) reproduces production exactly.
- **And it had to use the bot the APP started.** The first version built a second bot
  beside the running one, so the announcement hooks pushed into one object's
  `_announced` map while the taps landed on another's. `make_client` enters the
  lifespan, which starts the real bot - the tests now use that one and stub
  `getUpdates` with a 500 so its poll loop backs off instead of busy-spinning against
  respx.
- **A pytest fixture imported across test modules is a ruff F811 and a shadowing
  trap.** The real-hostd fixture moved to `conftest.py` (where two modules can share
  it by name) and gained an injectable clock, so a test can let an approval window
  actually lapse instead of reaching into engine internals.

## Self-reflected feedback

- **"Where does the trim cut?" is a question about consequence, not length.** The cap
  was implemented from the ledger's `cap-message-length-after-escaping-not-before`
  lesson - correctly - and still lost the undo line, because that lesson is about
  WHEN to trim and this bug was about WHERE. A cap over a rendered document needs to
  name which parts are load-bearing before choosing what to drop.
- **Two surfaces, one service, is worth the extra indirection.** The provider layer
  looked like ceremony while writing it; it is what made "the same enforcement" a
  fact to assert rather than a claim, and the review's hardest test
  (`test_telegram_approval_uses_the_same_enforcement`) compares the two refusals
  directly because there is one place they can come from.
