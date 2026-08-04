# Notes: Close the three open round-2 findings on the delivery contract

## What changes

Nothing an operator sees. This is a contract repair on the storage layer that
Lane 2's channel will be written against.

Before:

- Two overlapping passes over one channel both get `True` from
  `claim_delivery` (correct - both should send), both send, and then the
  SECOND `confirm_delivery` raises `LookupError` into the caller loop, because
  the row is already `confirmed` and the UPDATE matches nothing. The docstring
  says "no correct caller reaches it", which is false.
- `claim_delivery` carries a conflict-loser branch that cannot be reached
  (engine begins are `BEGIN IMMEDIATE`,
  `packages/core/src/scufris_core/engine.py:268`), is untested, and would raise
  `NoResultFound` if it ever were reached - its re-SELECT runs on the same
  snapshot that already answered `None`. Its docstring claims "the answer is
  true by construction".
- Two README paragraphs have orphan part-lines left by a reflow.

After:

- `confirm_delivery` is the exact mirror of `claim_delivery`: it accepts a row
  that is `claimed` OR already `confirmed`, and raises only when there is no
  row at all - i.e. only for a confirmation of something that was never
  claimed, which is the error the raise exists for. The overlapping pass
  completes on both sides.
- `claim_delivery` has no unreachable branch: the INSERT stays
  `on_conflict_do_nothing()` and the function returns `True` without consulting
  `rowcount`, so there is one write and no branch without a caller.
- Both README paragraphs wrap whole at 80 columns.

## Surfaces

| File | Why |
|---|---|
| `packages/chat/src/scufris_chat/store.py` | `confirm_delivery` gains the `confirmed`-tolerant path; `claim_delivery` loses the dead conflict-loser branch; both docstrings restated |
| `packages/chat/tests/test_chat_delivery.py` | new case: claim, claim, confirm, confirm over one channel - the overlapping pass nothing covers today |
| `packages/chat/src/scufris_chat/README.md` | section 5's claim/confirm paragraph and section 7's refusal sentence restate the contract; two paragraphs re-wrapped (R2.3) |
| `tasks/20260804-115319/DECISION.md` | section 6's closing mirror sentence (lines 126-128) is the text R2.1 falsified; the review asks for it to be corrected in place |
| `tasks/20260804-141639/DECISION.md` | new: records why tolerance was chosen over documenting the raise, and why the loser branch was collapsed rather than kept |

No signature changes, so `examples/chat_conversation.py` and
`packages/chat/src/scufris_chat/__init__.py` are untouched.

## Data and interfaces

Unchanged signatures; changed contracts.

```python
def claim_delivery(
    conn: Connection, conversation_id: str, channel: str, event_seq: int
) -> bool: ...
# True = send this (minted, or a claimed row nobody confirmed). False = confirmed.
# Unchanged. Raises LookupError only when the event does not exist.

def confirm_delivery(
    conn: Connection, conversation_id: str, channel: str, event_seq: int
) -> None: ...
# Now: no-op for an already-confirmed row. LookupError ONLY when no row exists.
```

`_delivery_state(conn, conversation_id, channel, event_seq) -> str | None`
already exists and is what the tolerant branch re-reads.

## Sketches

Illustrative, not the patch.

`confirm_delivery`:

```diff
     if not confirmed.rowcount:
-        raise LookupError(
-            f"channel {channel!r} has no claimed delivery of event {event_seq} "
-            f"of conversation {conversation_id!r} to confirm"
-        )
+        state = _delivery_state(conn, conversation_id, channel, event_seq)
+        if state is None:
+            raise LookupError(
+                f"channel {channel!r} has no delivery of event {event_seq} of "
+                f"conversation {conversation_id!r} to confirm"
+            )
+        # Already confirmed. The send this call is reporting did happen; the
+        # row already says so, and confirmed_at keeps the first return's time.
```

`claim_delivery`:

```diff
-        if minted.rowcount:
-            return True
-        # Another claimant got there between the read and the insert. ...
-        state = conn.execute(select(DeliveryRow.state).where(...)).scalar_one()
+        return True
```

New test, in `packages/chat/tests/test_chat_delivery.py`:

```python
def test_two_overlapping_passes_over_one_channel_both_complete(...):
    # claim A -> True, claim B -> True, confirm A -> ok, confirm B -> ok
    # and the row is confirmed exactly once, with A's confirmed_at
```

## Shape

```
  pass A                     delivery row                    pass B
    |                             |                            |
 claim -----> mint (claimed) -----+                            |
    |                             |<---------------- claim (re-claim, True)
  send                            |                          send
    |                             |                            |
 confirm ---> claimed->confirmed -+                            |
                                  |<-------------- confirm  ---+
                                  |   rowcount 0 -> re-read
                                  |   confirmed  -> no-op   (was: LookupError)
                                  |   no row     -> LookupError
```

The raise now fires on exactly one input: a confirm for a `(channel,
conversation, seq)` that was never claimed. That is the case with no FOREIGN
KEY behind it and the one the refusal was written for.

## Consequences and open questions

- The two-`True` claim and the tolerant confirm are one contract read from both
  ends: "should I send" is answered the same way at both calls, and a caller
  running overlapping passes needs no way to tell the cases apart. This is why
  tolerance beat documenting the raise - the alternative pushes a `try/except
  LookupError` into every channel Lane 2 onward writes, to catch a state that
  is not an error.
- Cost: `confirm_delivery` no longer catches a double-confirm in one pass. That
  was never a distinguishable error anyway - a second confirm of the same row
  is indistinguishable from the overlapping pass we now allow.
- `confirmed_at` keeps the FIRST confirmation's time, unlike `claimed_at`,
  which a re-claim restamps. The reason differs: `claimed_at` means "when the
  live attempt started", `confirmed_at` means "the send returned", and the
  earliest true answer is the one that stays true.
- Collapsing the loser branch means `on_conflict_do_nothing()` is now belt with
  no observable braces under this engine. It stays because it costs one clause
  and turns a foreign-engine conflict into a no-op instead of an
  `IntegrityError` mid-loop; the docstring will claim only that.
- Forecloses nothing. No lease, no reaper, no state accessor is exported, so a
  later lane that wants to tell "minted" from "re-claimed" apart still has to
  ask for it deliberately.
- Assumption recorded rather than asked: R2.1 and R2.2 each offered two
  sanctioned directions, and this picks tolerance for R2.1 and collapse-to-
  `return True` for R2.2. Both are the reviewer's own options; the plan can
  swap either without touching the other.
- Open: editing `tasks/20260804-115319/DECISION.md` in place cuts against
  "records are history", which round 2 itself flagged as a process signal on
  that task. The review explicitly asks for the correction, so the plan should
  amend section 6 with a dated round-2 note rather than a silent rewrite.
