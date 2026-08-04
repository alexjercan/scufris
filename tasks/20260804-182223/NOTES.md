# Understanding: host approval decoupling

## What changes

The decision half comes out of `HostApprovalService`. `hostctl` keeps propose,
apply, deny and audit - the things that talk to `hostd`. The write order
reverses so a crash is recoverable.

## Surfaces

- `packages/hostctl/src/scufris_hostctl/approvals.py` - 549 lines, and the
  carve epic called it complete.
- `approve()` at `:380-415`, and `_fire` at `:415` specifically.
- `scufris/telegram/approvals.py:79` - `_announced`, the `OrderedDict` being
  replaced.
- `chat`'s `delivery` table, for the real key.

## Data and interfaces

`approve(action_id, *, actor: str, acknowledge: str = "")` becomes
`approve(action_id, *, decision: OperatorDecision, acknowledge: str = "")`.

That is the security change in one line: `actor: str` is a string any caller can
fabricate, and `OperatorDecision` cannot be constructed outside `authorize`.

The idempotency key becomes `(channel, conversation_id, event_seq)` from
`delivery`, which survives a restart. `_announced` is capped at
`MAX_TRACKED_ACTIONS` and lives in memory, so it fails in two ways today: a
restart loses it, and a busy period evicts it.

## Sketches

```
  TODAY                                AFTER
    claim the decision (COMMIT)          write the conversation event (COMMIT)
    attach the run                                |
    _fire(on_decided)  <-- :415          apply via hostd
        |                                         |
    crash HERE?                          crash HERE?
        |                                         |
    the row says approved                the log says an operator approved
    the conversation NEVER hears it      hostd still holds it pending
    -> unrecoverable                     -> recoverable, replay the apply
```

## Shape

The guarantee to preserve, from the module's own docstring: "``apply`` is called
from exactly one place - ``approve`` below - so an action with no approval has no
route to execution, not because a check refuses it but because nothing else
calls it."

Moving the decision out must not weaken that. It does not: `approve()` stays the
only caller of `apply`, and its argument becomes unforgeable. The property is
strengthened, and that should be asserted rather than asserted-about.

## Consequences and open questions

- **Blocked on `20260804-182222`.** The signature cannot be written until the
  import path for `OperatorDecision` is decided. If option C wins there, this
  task changes shape substantially - `approve()` may not stay in `hostctl` at
  all.
- **Blocked on `20260804-141639`.** Round 2 of `115319`'s review falsified part
  of `confirm_delivery`'s docstring: a re-claim hands `True` back for a row it
  did not leave claimed, so two overlapping passes both send and the second
  confirm raises. This task writes a channel against that contract, so it
  should be written against the corrected one.
- **Open:** what happens to `_fire` and the three hook lists (`on_proposed`,
  `on_restored`, `on_decided`). They swallow listener exceptions deliberately -
  "a listener must not fail a decision" - which is right for a notification and
  wrong for a durable event. If the conversation event goes through a hook, it
  inherits the swallow. It probably should not be a hook at all.
- **Open:** whether `attach_run` moves too. It sits between the claim and the
  fire today, and the reversal has to place it somewhere deliberate.
- **Risk, recorded in the epic:** this re-opens 549 passing lines in a package
  the carve declared complete. It is the Lane 2 task most likely to overrun.
