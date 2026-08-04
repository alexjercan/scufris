# Close the three open round-2 findings on the delivery contract

- PRIORITY: 40
- TAGS: feature, v0.2.0, lane1, chat
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260801-154211
- DEPENDS ON: 20260804-115319

## Story

As the operator, I want the delivery contract to say what the code does, so
that Lane 2's channel is written against a true docstring and does not carry a
`LookupError` it was told could not happen.

## Steps

- [x] 1. R2.1 test first. Add
      `test_two_overlapping_passes_over_one_channel_both_complete` to
      `packages/chat/tests/test_chat_delivery.py`, driving one transaction each
      for claim A (`True`), claim B (`True`), confirm A, confirm B, and
      asserting `_delivery_rows` ends `[(TELEGRAM, id, seq, "confirmed")]`.
      Assert `confirmed_at` is the FIRST confirmation's stamp by reading the
      column before and after confirm B and comparing equal. Red on base with
      `LookupError: channel 'telegram' has no claimed delivery of event 1`,
      reproduced in scratch at `store.py:313`.
- [x] 2. R2.1 code. In `packages/chat/src/scufris_chat/store.py`,
      `confirm_delivery`: replace the bare `raise` under `if not
      confirmed.rowcount` with a re-read through the existing
      `_delivery_state(conn, conversation_id, channel, event_seq)` - `None`
      raises `LookupError` (message drops "claimed": "has no delivery of event
      ... to confirm"), anything else returns. Restate the docstring: the raise
      fires on exactly one input, a confirm for a key that was never claimed,
      and an already-`confirmed` row is a no-op that keeps the first
      confirmation's `confirmed_at`. Keep the `state == CLAIMED` guard on the
      UPDATE - it is what makes `confirmed_at` first-write-wins.
- [x] 3. R2.2. Same file, `claim_delivery`: delete lines 266-275 (the
      `minted.rowcount` test and the re-SELECT) and `return True` directly after
      the INSERT. `on_conflict_do_nothing()` stays; the docstring drops "the
      answer is true by construction" and claims only what it does - one write,
      and a conflict under a foreign engine's deferred begin is a no-op instead
      of an `IntegrityError` mid-loop.
- [x] 4. R2.3 plus the prose the contract change touches, in
      `packages/chat/src/scufris_chat/README.md`. Re-wrap whole the two
      paragraphs holding the orphan part-lines `checks what the` (line 147) and
      `recorded choice, not an` (line 410) - the review cited 213/238, which the
      file has since moved. Restate section 5's channel-pass paragraph (250-256)
      with the confirm mirror, and the `confirm_delivery` API-table row (line
      379) from "`LookupError` if nothing is sitting in `claimed`" to
      `LookupError` only when there is no delivery at all. Section 7's sentence
      (398-400) already says "a confirmation of something that was never
      claimed" and stays true unchanged - verify, do not rewrite.
- [x] 5. Records. `tasks/20260804-141639/DECISION.md` and the reciprocal
      `- STATUS: SUPERSEDED by tasks/20260804-141639/DECISION.md` on
      `tasks/20260804-115319/DECISION.md` are both written at plan time; section
      6's prose is history and stays as it is. Re-read the new record against
      what was actually built and correct it if the build diverged, then
      `tatr check`.
- [x] 6. Green the gates: `python -m pytest packages/chat/tests -q`, then
      `ruff check .`, `ruff format .`, `mypy .`, then the full
      `python -m pytest`.

## Definition of Done

- Two overlapping delivery passes over one channel both complete, and the row
  ends `confirmed` once with the first pass's `confirmed_at`
  (test: `test_two_overlapping_passes_over_one_channel_both_complete` in
  `packages/chat/tests/test_chat_delivery.py`)

  cmd: `python -m pytest packages/chat/tests -q`

- `confirm_delivery` still refuses a confirmation of something that was never
  claimed
  (test: the existing `test_delivery_requires_its_event` stays green unchanged)

  cmd: `python -m pytest packages/chat/tests -q`

- `claim_delivery` has no branch without a caller path or a test
  (manual: read `claim_delivery` end to end; every branch is `None` -> mint,
  `confirmed` -> `False`, `claimed` -> restamp and `True`, each covered by an
  existing case)
- Both README paragraphs wrap whole, and no line of section 5, 7 or 8 prose is
  a short part-line mid-paragraph
  (manual: read `README.md` lines 141-149, 250-256, 390-400 and 404-415 whole)
- `confirm_delivery`'s docstring, README section 5 and
  `tasks/20260804-115319/DECISION.md` section 6 all state the contract the
  tests assert
  (manual: read the three side by side; no remaining claim that a correct
  caller cannot reach the raise)
- The whole backend suite is green and the task records lint clean
  (test: nothing outside `packages/chat` moves, so a full run is the check that
  the contract change reached no other caller)

  cmd: `python -m pytest`

  cmd: `tatr check`

## Notes

- Source: `tasks/20260804-115319/REVIEW.md` round 2. All three were APPROVEd
  open as MINOR/NIT; none blocks the delivery table.
- `tasks/20260804-141639/NOTES.md` holds the understanding pass, including the
  shape diagram and the two rejected directions. This plan takes tolerance for
  R2.1 and collapse-to-`return True` for R2.2; both are the reviewer's own
  sanctioned options and either can be swapped without the other.
- Proof run on base: a scratch copy of Step 1's case fails at
  `packages/chat/src/scufris_chat/store.py:313` with the `LookupError`. Deleted
  after the check; Step 1 writes the real one.
- Suite runs under `nix develop`; `python -m pytest`, never bare `pytest`
  (`AGENTS.md`).
- No signature changes, so `examples/chat_conversation.py` and
  `packages/chat/src/scufris_chat/__init__.py` are untouched. The example's
  loop (line 135) is the caller the tolerant confirm exists for.

## Close-out

**What and why.** `confirm_delivery` is now the mirror of `claim_delivery`. The
UPDATE keeps its `state == claimed` guard, and when it matches nothing the
function re-reads through `_delivery_state`: `None` raises, anything else
returns. So the raise fires on exactly one input - a confirm for a key that was
never claimed - and two overlapping passes over one channel both complete, with
`confirmed_at` holding the FIRST confirmation's stamp. `claim_delivery` lost the
`minted.rowcount` test and the re-SELECT behind it; it returns `True` straight
after the INSERT. `on_conflict_do_nothing()` stayed, and both docstrings now
claim only what the code does. README section 5 gained the confirm-mirror
paragraph, the API table's `confirm_delivery` row states the narrowed raise, and
sections 7 and 8 were re-wrapped whole.

**Alternatives.** `tasks/20260804-141639/DECISION.md` records the four, and the
build took the two it chose: tolerate the confirm rather than document the raise
(which would push a `try/except` into every Lane 2 channel to catch a state that
is not an error), and collapse the claim rather than repair an unreachable
conflict-loser branch.

**Difficulties.** The plan located R2.3's two orphan part-lines at README 147
and 410 and warned the file had moved. Only 410 was still an orphan; the
paragraph around 147 already greedy-wraps at 80. A width scan over the whole
file found the real second one at 393, inside the same section 7 paragraph the
plan named, so both paragraphs the review meant were re-wrapped whole. Sections
5, 7 and 8 now have no short mid-paragraph line; the four candidates left in the
file are in sections 1, 2 and 6, where the next word genuinely does not fit.

**Evidence.** `test_two_overlapping_passes_over_one_channel_both_complete` was
red on base at `store.py:313` with the exact `LookupError` the plan predicted,
green after Step 2. `test_delivery_requires_its_event` stayed green unchanged,
which is what holds the narrowed raise honest. `packages/chat/tests` 22 passed;
full suite 1139 passed, 1 skipped; `ruff check`, `ruff format`, `mypy` (250
files) and `tatr check` all clean.

**Manual criteria.** `claim_delivery` read end to end: three branches, `None` ->
mint -> `True`, `confirmed` -> `False`, `claimed` -> restamp -> `True`, covered
by `test_delivery_is_idempotent_on_replay`,
`test_a_claimed_delivery_is_pending_until_confirmed` and the new case. For the
side-by-side criterion: `confirm_delivery`'s docstring and README section 5 both
state the contract the tests assert, and `tasks/20260804-115319/DECISION.md`
carries the reciprocal `SUPERSEDED by` header. Its section 6 prose still reads
"no correct caller reaches that" - Step 5 is explicit that section 6 is history
and stays as it is, so the supersession header plus this record's opening
quotation of that sentence is how the falsified clause is retired. Flagged for
review as the one place the DoD bullet and Step 5 pull in different directions.

**Reflection.** The plan's line numbers were stale in a way the plan itself
predicted; the cheap defence was a width scan rather than trusting either the
cited lines or the reviewer's memory of them. Worth reaching for whenever a
finding is about wrapping.
