# Review: Close the three open round-2 findings on the delivery contract

- TASK: 20260804-141639
- BRANCH: feature/delivery-contract-round2

## Round 1

- REVIEWER: out-of-context (two lanes: correctness/security/concurrency,
  spec/design/standards/docs)
- VERDICT: APPROVE

- [ ] R1.1 (MINOR) packages/chat/src/scufris_chat/store.py:233 - Step 3 deleted
  the conflict loser's re-SELECT, so under a foreign engine's deferred begin a
  claimant that read `None`, lost the INSERT race, and whose winner then
  confirmed now answers `True` for a `confirmed` row - a duplicate send. The
  first docstring paragraph still promises "``False`` only for a ``confirmed``
  row, which is what makes a replay of a completed delivery a no-op at the
  STORAGE layer" without qualification, while the last paragraph qualifies only
  the `IntegrityError` case. Add one clause to the first paragraph: the
  completed-delivery no-op is guaranteed under `scufris_core.engine`'s immediate
  begin, and a conflict loser under a deferred begin answers `True` without
  re-reading. Unreachable in-tree - no non-immediate engine exists - and the
  package already prefers a duplicate to a loss, so it is a precision gap in the
  contract prose, not a behavior defect.
  - Response:
- [ ] R1.2 (MINOR) tasks/20260804-115319/DECISION.md:127 - the DoD's side-by-side
  manual criterion asks that the docstring, README section 5 and that record's
  section 6 leave "no remaining claim that a correct caller cannot reach the
  raise". Section 6 still reads "no correct caller reaches that, because every
  one gates its send on a `True` claim", and the diff changed only the record's
  `STATUS:` header. Step 5 ("section 6's prose is history and stays as it is")
  and the DoD bullet pull in opposite directions, and NOTES.md had proposed a
  third handling; the close-out discloses the conflict honestly rather than
  ticking it. Append one dated line under that sentence - "Round 2
  (20260804-141639) falsified the last clause; see
  `tasks/20260804-141639/DECISION.md`." - which retires the clause where a Lane 2
  author reads it and stays append-only.
  - Response:
- [ ] R1.3 (NIT) packages/chat/src/scufris_chat/store.py:313 - two nested `if`s
  with no `else` on either, and the inner one gates on `is None`, accepting any
  non-null state as "already delivered". Only `confirmed` can reach here under
  the `ck_delivery_state` CHECK, so it is safe today. Collapse both into
  `if not confirmed.rowcount and _delivery_state(conn, conversation_id, channel,
  event_seq) != DeliveryState.CONFIRMED.value:` - one level less, and a future
  third state becomes a refusal rather than a silent success. Step 2 specified
  the `is None` shape literally, so this is a departure from the approved plan,
  not a defect in following it.
  - Response:
- [ ] R1.4 (NIT) packages/chat/tests/test_chat_delivery.py:337 - first-write-wins
  on `confirmed_at` is pinned by comparing two wall-clock `time.time()` stamps
  read in separate transactions. It does go red when the `state == CLAIMED` guard
  is deleted, so it is not a test that cannot fail, but the signal is a clock
  delta rather than a controlled value. Monkeypatch `store.time.time` to return
  two distinct fixed values and assert the exact first one.
  - Response:

Process signal: Step 5 and the Definition of Done disagreed about
`tasks/20260804-115319/DECISION.md` section 6, and NOTES.md proposed a third
handling again. The implementer arbitrated at build time and flagged it in the
close-out; the plan gate should have resolved it. Both lanes raised this
independently.

Process signal: Step 4 anchored a re-wrap to README line numbers (147, 410) the
plan itself predicted were stale. Only 410 was still an orphan; a width scan
found the real second one at 393. Naming a paragraph by its opening words
outlasts a line number.

Verified by the recording pass, independently of both lanes: full
`python -m pytest` exit 0 (1 skipped), `packages/chat/tests` 22 passed,
`ruff check .`, `ruff format --check .` (250 files), `mypy .` (250 files) and
`tatr check` all clean in `nix develop`. Red-on-base re-derived directly by
restoring `master`'s `store.py` in the worktree and running the new case: it
failed with `LookupError: channel 'telegram' has no claimed delivery of event 1`
at `store.py:313`, exactly the reproduction TASK.md claims, and the tree was
restored clean afterwards. `test_delivery_requires_its_event` is untouched by
the diff. Doc sweep: no `no correct caller`, `true by construction`,
`sitting in claimed` or `has no claimed` text survives anywhere outside
`tasks/`. Both DECISION.md records carry the reciprocal supersede link.

Pending user checks, none blocking: the three `manual:` proofs - reading
`claim_delivery` end to end, reading README lines 141-149, 250-256, 390-400 and
404-415 whole, and reading the docstring, README section 5 and
`tasks/20260804-115319/DECISION.md` section 6 side by side. Both lanes report
the first two as satisfied and the third as the subject of R1.2.

Inspection commands:

```bash
cd "$(sprout show feature/delivery-contract-round2)"
git diff master...HEAD
nix develop --command python -m pytest packages/chat/tests -q
nix develop --command mypy .
```
