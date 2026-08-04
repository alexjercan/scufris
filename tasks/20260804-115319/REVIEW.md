# Review: Deliver chat events to every channel exactly once

- TASK: 20260804-115319
- BRANCH: feature/chat-delivery

## Round 1

- REVIEWER: out-of-context (three lanes: behavior/proofs,
  correctness/security/persistence/concurrency, design/standards/docs)
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (BLOCKER) packages/chat/src/scufris_chat/store.py:186 -
  `claim_delivery` returns `False` for ANY existing row, including one still
  `claimed`, so the crash-mid-send case this whole table exists for is never
  retried. The documented channel pass - `packages/chat/src/scufris_chat/README.md`
  section 5 and `examples/chat_conversation.py:56` "the claim's answer is what
  drives the send" - reads `pending_events`, gets the claimed event back, calls
  `claim_delivery`, gets `False`, and skips. The row stays `claimed` and pending
  FOREVER and the operator never sees the question. That is the "silently loses
  the question" failure DECISION.md section 2 rejects, and it falsifies README
  section 5's "a crash between the send and the confirm re-sends once on
  restart. Duplicate, not lost." and the DoD's "retried rather than lost". The
  smoking gun is line 188: the SELECT asks for `DeliveryRow.state` and the
  result is discarded. Reproduced against the branch by driving the example's
  own `deliver()` loop after a committed-but-unconfirmed claim - three restarts
  in a row printed `pending=[1] sent=[]`. Change: branch on the fetched state -
  return `True` for an existing `claimed` row (a re-claim, refreshing
  `claimed_at`), `False` only for `confirmed`.
  - Response: fixed - `claim_delivery` now branches on the fetched
    state: `True` for a row it minted AND for an existing `claimed` one (a
    re-claim, restamping `claimed_at`), `False` only for `confirmed`. Verified
    red-for-this-reason: with the old `if state is not None: return False`
    restored, R1.2's rewritten test fails `assert [] == [1]`.
- [x] R1.2 (MAJOR) packages/chat/tests/test_chat_delivery.py:210 -
  `test_a_claimed_delivery_is_pending_until_confirmed` is the test that should
  have caught R1.1, and cannot: after the simulated crash it calls
  `confirm_delivery` directly (line 238), never re-running the claim path a
  restarting channel actually runs. It proves "a claimed row is pending" but not
  its own docstring's "a crash between the send and the confirm is retried, not
  lost", which is why the suite is green while the central promise is broken.
  Rewrite the recovery half to drive the SAME loop the example documents
  (`pending_events` -> `claim_delivery` -> send -> `confirm_delivery`) and
  assert the event is sent exactly once more and the row ends `confirmed`.
  - Response: fixed - the recovery half is now a `restart()` closure driving
    `pending_events` -> `claim_delivery` -> send -> `confirm_delivery`, the same
    loop the example documents, asserting the event is sent exactly once more,
    the row ends `confirmed`, and a further restart sends nothing.
- [x] R1.3 (MAJOR) tests/test_db_schema.py:246 - the shipped `ck_delivery_state`
  text is never asserted against `DELIVERY_STATE_VALUES`, so
  `models.py:64`'s claim that the constraint is "rendered from the enum ... so a
  third `DeliveryState` cannot land with the constraint still naming two" is
  false for an operator's migrated database. The sibling
  `test_migrated_actor_check_lists_exactly_the_declared_kinds` exists in this
  same file for exactly this hole and records the reason: Alembic's
  `compare_metadata` does not diff CHECK constraints, so a third state would be
  green in the models, in the `Base.metadata`-built package tests AND in
  `test_schema_has_no_pending_autogenerate_diff`, and raise `IntegrityError`
  only in production. Add the mirror test reading `sqlite_master` for `delivery`
  and asserting the listed values equal `tuple(s.value for s in DeliveryState)`.
  - Response: fixed - added
    `test_migrated_delivery_check_lists_exactly_the_declared_states`, the mirror
    of the actor test: reads `delivery`'s DDL from `sqlite_master` and asserts
    the listed values equal `tuple(state.value for state in DeliveryState)`.
    `DeliveryState` is now exported from the facade so the test does not import
    a private module.
- [x] R1.4 (MINOR) examples/chat_conversation.py:60 - `deliver()`'s docstring
  calls itself "the shape every channel has", but it performs the send INSIDE
  `database.transaction()`, between the claim and the confirm. That is the one
  arrangement the two-state design exists to avoid, and it is the shape Lane 2
  will copy. Once R1.1 is fixed, restructure it the way the design means: claim
  in one unit of work, send, confirm in a second.
  - Response: fixed - `deliver()` reads its backlog, then per event claims and
    commits, sends, and confirms in a second transaction. The docstring now
    states why the send sits between two units of work.
- [x] R1.5 (MINOR) packages/chat/src/scufris_chat/models.py:184 - `channel` has
  no non-empty CHECK, though the sibling `_ACTOR_AGENT_ID_CHECK`
  (`models.py:60`) was given `actor_agent_id <> ''` on the stated reasoning that
  a repair INSERT with an uninterpolated variable produces `''` more readily
  than a name. A `''` channel is a distinct primary key, so the delivery lands
  under a channel nothing polls and the real channel still sees the event as
  pending. Add `channel <> ''` to `__table_args__` and to the revision.
  - Response: fixed - `ck_delivery_channel` (`channel <> ''`) added to
    `__table_args__` and to revision `53aaa107ce2d`, with an `IntegrityError`
    case in `test_migration_creates_the_delivery_table`.
- [x] R1.6 (MINOR) packages/chat/src/scufris_chat/store.py:236 - the second
  SELECT exists only to make an already-`confirmed` row a silent no-op instead
  of a `LookupError`. No caller reaches it (every caller gates
  `confirm_delivery` behind a `True` claim), no test covers it, and the choice
  is recorded in TASK.md's close-out but not in DECISION.md. Either delete lines
  236-247 and raise whenever `rowcount` is 0, or add the double-confirm test
  that makes the branch a requirement.
  - Response: fixed by deletion - `confirm_delivery` now raises whenever the
    UPDATE matches no `claimed` row. The message covers both the never-claimed
    and the already-confirmed case, and the docstring's timestamp-preservation
    claim went with the branch.
- [x] R1.7 (MINOR) tasks/20260804-115319/TASK.md:189 - the close-out's Evidence
  states "`nix flake check` - all 8 checks passed". The flake declares SIX
  checks (`flake.nix`: ruff, ruff-format, mypy, pytest, records, filesize) and
  the run reports "running 6 flake checks". The substance is right - I re-ran it
  and all checks passed - but the number is one no rig produced. Correct it to
  6. The other recorded counts (11 in `packages/chat/tests`, 25 across the two
  migration modules) were re-run and are accurate.
  - Response: fixed - corrected to 6. This round's `nix flake check` reports
    "running 6 flake checks" and all 6 passed.
- [x] R1.8 (MINOR) examples/chat_conversation.py:80 - the comment above
  `Base.metadata.create_all` still says "`metadata` holds exactly
  `conversation` and `event`". This diff added a third declared table to that
  same metadata and the example now creates it. Name all three.
  - Response: fixed - the comment names `conversation`, `event` and `delivery`.
- [x] R1.9 (MINOR) packages/chat/tests/test_chat_delivery.py:50 - the `database`
  fixture docstring points at
  `tests/test_db_migrations.py::test_migration_creates_the_delivery_table`; this
  diff put that test in `tests/test_db_schema.py`. Repoint it.
  - Response: fixed - repointed at `tests/test_db_schema.py`.
- [x] R1.10 (MINOR) AGENTS.md:18 - the `packages/*/pyproject.toml` row still
  reads "`packages/chat` -> `scufris_chat`, the conversation and its events"
  while line 22, four rows below, was updated in this same diff to name the
  delivery table. Add delivery to line 18.
  - Response: fixed - line 18 now reads "the conversation, its events and their
    per-channel delivery".
- [x] R1.11 (NIT) packages/chat/src/scufris_chat/store.py:172 vs :251 -
  `claim_delivery(conn, channel, conversation_id, ...)` against
  `pending_events(conn, conversation_id, channel)`. The close-out flags this
  itself and defers it to Lane 2. Two adjacent functions on one public surface
  with swapped ids is a foot-gun; align both on
  `(conn, conversation_id, channel, ...)` now, while there is no caller to
  migrate.
  - Response: fixed - `claim_delivery` and `confirm_delivery` now take
    `(conn, conversation_id, channel, event_seq)`, matching `pending_events`.
    All 18 call sites, the example, and the README surface table updated.
- [x] R1.12 (NIT) packages/chat/src/scufris_chat/store.py:184 - `claim_delivery`
  runs `_require_event` BEFORE checking for an existing row, so once retention
  arrives (README section 7 flags it as coming) a replay of an already-confirmed
  delivery raises `LookupError` instead of being the promised no-op. Check for
  the existing row first; require the event only on the mint path.
  - Response: fixed - the existing row is read first; `_require_event` runs only
    on the mint path, so a replay of a confirmed delivery stays a no-op once
    retention removes the event underneath it.
- [x] R1.13 (NIT) packages/chat/src/scufris_chat/store.py:181 - the docstring
  asserts "the caller's transaction, whose begin is immediate" as an unchecked
  precondition. It does hold - re-derived independently at
  `packages/core/src/scufris_core/engine.py:268` - but a connection from any
  other engine turns the read-then-insert into an `IntegrityError` on the loser
  rather than a `False`. `INSERT ... ON CONFLICT DO NOTHING` with the answer
  read off `rowcount` would make the function true by construction, and folds
  neatly into R1.1's fix.
  - Response: fixed - the INSERT is `sqlite_insert(...).on_conflict_do_nothing()`
    with the answer read off `rowcount`, following `scufris/scheduler.py:133`.
    A loser re-reads the winner's state and answers it like any other
    pre-existing row, so the function no longer rests on the immediate begin.
- [x] R1.14 (NIT) packages/chat/src/scufris_chat/models.py:46 -
  `DELIVERY_STATE_VALUES` has one reader (line 68). `ACTOR_KIND_VALUES` earns
  its name with readers across modules. Inline the generator into
  `_DELIVERY_STATE_CHECK` - unless R1.3's test makes it a second reader, in
  which case keep it.
  - Response: fixed by inlining - `DELIVERY_STATE_VALUES` is gone and
    `_DELIVERY_STATE_CHECK` renders from `DeliveryState` directly. R1.3's test
    reads the exported enum rather than the constant, so it did not become a
    second reader.
- [x] R1.15 (NIT) packages/chat/src/scufris_chat/README.md:204,222 - two new
  prose lines run to 94 and 101 columns where the rest of the file's prose wraps
  at 80 (the other long lines are table rows). Reflow.
  - Response: fixed - both lines reflowed to 80.

Process signal: Step 8's growth crossed the 900-line test cap and forced a file
split, a new `tests/conftest.py` fixture and a `scufris/README.md` edit. That
was foreseeable at plan time from `test_db_migrations.py`'s size; a separate
split task would have kept this branch to the delivery table.

Process signal: Step 8's literal text and two Definition-of-Done entries were
EDITED on this branch rather than left as planned with the deviation recorded
only in the close-out, so the planned step now reads as if the split was
planned. The close-out does disclose it, which is why this is a signal and not a
finding.

Process signal: the README and the example document the exact usage pattern that
fails R1.1, so the defect is in the specified contract, not only in the code -
Lane 2 would have copied it. `DeliveryState` and `DELIVERY_STATE_VALUES` are
also unexported, so a caller has no supported way to tell "already confirmed"
from "claimed, resend", which is why R1.1 has no caller-side workaround.

Verified by the recording pass, independently of the lanes: `nix flake check`
run end to end, all 6 checks passed. The DoD command
`pytest packages/chat/tests tests/test_db_migrations.py tests/test_db_schema.py
tests/test_examples.py -q` is green (45 passed); `packages/chat/tests` is 11 and
the two migration modules are 25, as recorded. The `BEGIN IMMEDIATE` claim
`claim_delivery`'s docstring rests on was re-derived at
`packages/core/src/scufris_core/engine.py:268`. R1.1 was reproduced from a
scratch script driving the example's own loop, not read off the code. The
`test_db_migrations.py` / `test_db_schema.py` split was checked to be a pure
move with no assertion lost, and is sanctioned by AGENTS.md's over-the-cap rule
with the allowlist ratchet untouched - it is not a scope finding. The migration
matches the models column for column, `down_revision` chains onto
`18c9104709b8`, the chain is linear and single-headed, downgrade round-trips,
and both CHECKs plus the composite PK are ENFORCED, not merely parsed.

No `manual:` proofs are pending on this task.

## Round 2

- REVIEWER: out-of-context
- VERDICT: APPROVE

All fifteen round-1 findings are confirmed fixed against the branch as it
stands and are ticked above. R1.13's fix landed as written - the INSERT is
`on_conflict_do_nothing()` with the answer off `rowcount` - but the branch it
adds does not deliver the property the response claims for it, which is R2.2
rather than an untick.

The three findings below are regressions of the round-1 fixes. None blocks.

- [ ] R2.1 (MINOR) packages/chat/src/scufris_chat/store.py:257 -
  `confirm_delivery`'s docstring ("No correct caller reaches it - every one
  gates its send behind a `True` from `claim_delivery`, which hands back only a
  row it left `claimed`") is falsified by R1.1's and R1.6's fixes taken
  together. R1.1 made `claim_delivery` answer `True` for a `claimed` row it did
  NOT mint, and R1.6 removed the tolerance for an already-`confirmed` row, so
  two overlapping passes over one channel both get `True`, both send, and the
  second `confirm_delivery` raises `LookupError` into an otherwise-correct
  caller loop. Reproduced on the branch from the facade: `pass A claim: True /
  pass B claim (re-claim): True / pass A confirm: ok / pass B confirm RAISED:
  channel 'tg' has no claimed delivery of event 1 ...`. Before round 1 the
  second claim answered `False` and skipped. Correct the docstring and
  `DECISION.md:126-128` to state the contract that now holds - a caller that
  may run two passes at once must tolerate the raise - or make
  `confirm_delivery` tolerant of an already-`confirmed` row and record why.
- [ ] R2.2 (MINOR) packages/chat/src/scufris_chat/store.py:228 - the
  conflict-loser branch R1.13 introduced is unreachable, untested, and would not
  answer if it were reached. Engine begins are immediate
  (`packages/core/src/scufris_core/engine.py:268`), so no second writer commits
  between `_delivery_state` and the INSERT. Under the deferred-begin engine the
  docstring invokes ("a connection from any other engine"), the loser takes
  `OperationalError` on the INSERT rather than `rowcount == 0`; and were
  `rowcount == 0` reached, the re-SELECT at line 228 runs on the same snapshot
  that already answered `None`, so `scalar_one()` raises `NoResultFound`. The
  docstring's "the answer is true by construction" is therefore not delivered.
  Either drop lines 223-232 back to `return True` and restore the honest
  precondition, or keep the INSERT and correct the docstring to claim only what
  it does - one write instead of a read-then-write.
- [ ] R2.3 (NIT) packages/chat/src/scufris_chat/README.md:213,238 - R1.15's
  reflow left two orphan part-lines mid-paragraph: `checks what the` and
  `recorded choice, not an`. Re-wrap both paragraphs whole rather than the
  edited line alone.

Process signal: TASK.md Steps 4 and 6 still spell
`claim_delivery(conn, channel, conversation_id, event_seq)` while R1.11 shipped
`(conn, conversation_id, channel, event_seq)`. That is correct - records are
history and the round-1 close-out discloses it - but it is the mirror image of
round 1's signal about Step 8's text being edited in place. The same record now
does both, so the branch has no single rule for what a superseded step looks
like.

Process signal: `nix flake check`'s "running N flake checks" line counts the
UNCACHED checks, not the declared ones. A cold run on this branch printed
`running 8 flake checks` and the next run printed `running 0`, both ending
`all checks passed!`. The close-out's `all 6 checks passed` is right - `flake.nix`
declares six - but the parenthetical quoting the output as `"running 6 flake
checks"` is a number no run reproduces. Not a finding, because R1.7 asked for
the declared count and got it; worth knowing before the next record quotes that
line as evidence.

Verified by the recording pass, independently of the out-of-context reviewer:
`nix flake check` run cold, all checks passed. The DoD command
`pytest packages/chat/tests tests/test_db_migrations.py tests/test_db_schema.py
tests/test_examples.py -q` is green at 46, the count the close-out records.
`examples/chat_conversation.py` exits 0 and prints both channels sending events
1 and 2 and then replaying and sending nothing. R1.1's fix was re-pinned by
mutation on a copy OUTSIDE the worktree - restoring `if state is not None:
return False` fails `test_a_claimed_delivery_is_pending_until_confirmed` at
`assert [] == [1]`, which is R1.2's rewritten test doing exactly the job R1.2
asked of it. The mutation had to be run with `PYTHONPATH` pointed at the copy:
the dev environment's editable install resolves `scufris_chat` back to the
worktree, so a mutated copy tested naively passes and proves nothing. R2.1 was
reproduced from the exported facade, not read off the code.

No `manual:` proofs are pending on this task.

## Inspection commands

```bash
cd "$(sprout show feature/chat-delivery)"
nix flake check
nix develop -c python -m pytest packages/chat/tests tests/test_db_migrations.py \
    tests/test_db_schema.py tests/test_examples.py -q
```
