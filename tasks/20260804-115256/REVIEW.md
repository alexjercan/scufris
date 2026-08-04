# Review: Record the chat conversation and event tables with typed actors

- TASK: 20260804-115256
- BRANCH: feature/chat-conversation-events

## Round 1

- REVIEWER: out-of-context (three lanes: behavior/proofs,
  correctness/security/concurrency, design/standards/docs)
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) pyproject.toml:30 - `scufris/db/migrations/env.py:27` makes
  the root distribution import `scufris_chat`, but `scufris-chat` is in neither
  `[project.dependencies]` nor `[tool.uv.sources]`. `uv.lock`'s `scufris`
  package lists `scufris-core`, `-host`, `-hostctl`, `-hostd` and no chat, so
  the build's declaration of membership contradicts the `scufris ->
  scufris_chat` edge the diff added to `DECLARED_GRAPH`. It passes in-repo only
  because `uv sync` installs every workspace member; an installed root wheel
  raises `ModuleNotFoundError` at `alembic upgrade head`, which is the one code
  path this dependency exists for. `scufris-hostctl` was added to the same list
  for exactly this reason when `env.py` first imported it. Add `"scufris-chat"`
  to `[project.dependencies]`, `scufris-chat = { workspace = true }` to
  `[tool.uv.sources]` (line 49-52), and re-run `uv lock`.
  - Response: fixed in round 2. `"scufris-chat"` added to `[project.dependencies]` with a comment naming the `env.py` import as the reason, `scufris-chat = { workspace = true }` added to `[tool.uv.sources]`, `uv lock` re-run - `uv.lock`'s `scufris` package now lists it in both `dependencies` and `requires-dist`. The finding is right that in-repo green proved nothing here: `uv sync` installs every member regardless.
- [x] R1.2 (MAJOR) packages/chat/src/scufris_chat/models.py:84 - the actor
  invariant is only HALF in the schema. `actor_kind` is CHECK-constrained but
  nothing pins `actor_agent_id` to agree with it, so a row with
  `actor_kind='operator'` and a non-null `actor_agent_id` is schema-legal.
  `store._record` (`store.py:154`) then rebuilds an `Actor` and
  `__post_init__` raises, so ONE such row makes `read_transcript` raise
  `ValueError` for the WHOLE conversation, not just that event - the transcript
  becomes unreadable through the only API that reads it. Reproduced on a real
  database: the INSERT is accepted, and the next `read_transcript` raises
  `a operator actor takes no agent id, got 'smuggled'`. The reachability
  argument is the diff's own: README section 3 names the hand-written INSERT -
  a migration, a repair session, a later store - as exactly the attacker the
  CHECK is there to meet, and for this half of the rule it does not. The seq
  invariant got its schema backstop (`uq_event_conversation_seq`); this one
  should too. Add `CheckConstraint("(actor_kind = 'agent') = (actor_agent_id IS
  NOT NULL)", name="ck_event_actor_agent_id")` to `EventRow.__table_args__` and
  the matching `sa.CheckConstraint` to revision `18c9104709b8`, and assert the
  rejected INSERT in `test_migration_creates_the_chat_tables`.
  - Response: fixed in round 2. `ck_event_actor_agent_id` (`(actor_kind = 'agent') = (actor_agent_id IS NOT NULL)`) added to `EventRow.__table_args__` and to revision `18c9104709b8`. Both rejected INSERTs - `operator` with an id, `agent` without one - are asserted in `test_migration_creates_the_chat_tables` and in `test_actor_must_be_a_known_kind`, with the accepted `agent`+id row asserted alongside so the constraint cannot pass by rejecting everything. Sabotaged both ways: removing it from the revision fails the migration test, removing it from the model fails the package test. The docstring now records WHY half a rule was worse than none - one bad row makes the whole transcript raise.
- [x] R1.3 (MINOR) scufris/db/migrations/versions/18c9104709b8_chat_conversation_and_event.py:45
  - the revision's CHECK text is a hardcoded string listing the four kinds,
  and nothing compares it to `ACTOR_KIND_VALUES`. `models.py:28` renders the
  metadata constraint from the enum so the two "cannot drift", but that closes
  only the model half: Alembic's `compare_metadata` does not diff CHECK
  constraints (verified - widening the metadata CHECK to a fifth value against
  a fresh head database returns an empty diff), and `packages/chat/tests` builds
  its tables from `Base.metadata`. A fifth `ActorKind` would therefore be green
  across the entire suite, including
  `test_schema_has_no_pending_autogenerate_diff`, and raise `IntegrityError`
  only on a migrated operator database. Add a test that reads the CHECK text
  from `sqlite_master` on a migrated database and asserts it lists exactly
  `ACTOR_KIND_VALUES`, next to `test_migration_creates_the_chat_tables`.
  - Response: fixed in round 2. `test_migrated_actor_check_lists_exactly_the_declared_kinds` reads `event`'s DDL from `sqlite_master` on a migrated database and asserts the listed kinds equal `ActorKind`'s. Checked that it discriminates: adding a fifth member to `ActorKind` fails it and `test_actor_must_be_a_known_kind`, and nothing else. It asserts against `ActorKind` rather than `ACTOR_KIND_VALUES` because the root suite reaches the package through its facade only, and the facade exports the enum.
- [x] R1.4 (MINOR) packages/chat/src/scufris_chat/store.py:67 - `append_event`
  accepts any `conversation_id` and silently mints a phantom transcript.
  Reproduced: `append_event(conn, "no-such-conversation", ...)` returns
  `event_seq=1` and `read_transcript` reads it back. This is asymmetric with
  `causing_event`, which raises `LookupError` for the identical dangling
  reference and justifies it at `store.py:130-134` with "there are no FOREIGN
  KEYs here ... nothing at the schema level stops a caller passing an id that is
  not an event's". The same sentence applies to `conversation_id`. Select the
  conversation row inside the caller's transaction and raise `LookupError` when
  it is absent.
  - Response: fixed in round 2. `append_event` selects the conversation row inside the caller's transaction and raises `LookupError` when it is absent; `test_appending_to_an_unknown_conversation_raises` covers it and asserts the transcript stays empty. The asymmetry the finding names was the whole argument, and the docstring now carries it.
- [x] R1.5 (MINOR) README.md:33 - the "Where to read more" table has a row per
  package README (host, hostd, hostctl) and gained none for chat. Add a row
  after line 33 pointing at
  [`packages/chat/src/scufris_chat/README.md`](../../packages/chat/src/scufris_chat/README.md).
  - Response: fixed in round 2. Row added after the hostctl row, in package order.
- [x] R1.6 (MINOR) scufris/README.md:479 - section 8's Module map enumerates
  every workspace member and AGENTS.md declares it the source of truth for that
  map. It is now stale by one: the diff's own comment at
  `tests/test_package_boundaries.py:214` says "The six members that exist
  today". Add a `packages/chat` -> `scufris_chat` row to that table.
  - Response: fixed in round 2. `packages/chat` -> `scufris_chat` row added to section 8's Module map.
- [x] R1.7 (MINOR) CHANGELOG.md:8 - AGENTS.md:87 requires a `CHANGELOG.md`
  entry for a notable change, and the `[Unreleased]` section's existing entry is
  the analogous hostctl carve. A sixth distribution plus a shipped migration
  creating two operator-database tables is the same class of change. Add an
  `### Added` bullet under `[Unreleased]` naming `packages/chat`/`scufris-chat`,
  the `conversation` and `event` tables, and revision `18c9104709b8`.
  - Response: fixed in round 2. `### Added` entry under `[Unreleased]` naming the distribution, both tables, the actor kinds and revision `18c9104709b8`, and stating that no operator-visible surface reads a conversation yet.
- [x] R1.8 (NIT) packages/chat/src/scufris_chat/actors.py:83 - `agent_id or
  None` means `"operator:"` and `"system:"` parse as valid actors, while
  `"operator:alex"` correctly raises. The separator with an empty id is a
  malformed wire form, not a bare kind. Raise when the separator is present and
  the id is empty.
  - Response: fixed in round 2. `parse` now keeps the separator from `partition` and raises when it is present with an empty id. `operator:` and `agent:` are both asserted to raise.
- [x] R1.9 (NIT) packages/chat/src/scufris_chat/store.py:139 - `causing_event`
  matches `EventRow.id` across the whole table, unscoped by conversation, so a
  `causation_id` copied from another thread resolves and reads as this
  conversation's cause. Defensible - the id is a globally unique primary key and
  the event returned is genuinely the causing one - so this is a choice, not a
  bug. If causation is meant to be intra-conversation, add
  `EventRow.conversation_id == event.conversation_id` to the `where` and assert
  the cross-conversation case in
  `test_causation_resolves_to_the_causing_event`; otherwise say so in the
  docstring, which currently addresses only the dangling case.
  - Response: fixed in round 2, by scoping rather than by documenting. The lookup now carries `EventRow.conversation_id == event.conversation_id`, and the cross-conversation case is asserted in `test_causation_resolves_to_the_causing_event` against a real event in another thread. Causation is a claim about the transcript the event is IN; unscoped it resolved to a real event and read as this conversation's cause, which is worse than the dangling case the code already refused.
- [x] R1.10 (NIT) packages/chat/src/scufris_chat/actors.py:85 - `Actor.render`
  has no production caller: the store persists `actor_kind` and
  `actor_agent_id` as two columns, so the wire form is never written. Its only
  uses are the example's display padding (`examples/chat_conversation.py:86`)
  and the round-trip assertions at
  `packages/chat/tests/test_chat_events.py:174-176`; no Step names a render
  direction, only the parse. Keep it if Lane 6's delivery work is known to need
  a single-column wire form, otherwise delete it with those three test lines and
  the README:119 mention.
  - Response: fixed in round 2 by deleting it. Lane 6 is not known to need a single-column wire form, and YAGNI decides it: no production caller, no Step naming a render direction. Gone with the three round-trip assertions and the README mention; the example formats the actor itself. The module docstring now says the parse is one-way ON PURPOSE, so the symmetry is not restored by reflex later.

Verification. Full suite green under `nix develop`: `python -m pytest`,
`ruff check .`, `ruff format --check .` (242 files), `mypy .` (242 files),
`uv lock --check`, `tatr check`. All six DoD proofs run and pass on their stated
criteria, including `python examples/chat_conversation.py` (exit 0). Revision
`18c9104709b8` is the single head on `4119562b5fd9` and its DDL matches
`models.py` column for column; `downgrade` drops in the correct order. No
existing test was deleted or weakened. Every finding above was reproduced by the
recording pass, not taken on a lane's word.

The concurrency claim was re-derived independently rather than accepted: running
this store's own code over an engine WITHOUT the `BEGIN IMMEDIATE` listener
(`packages/core/src/scufris_core/engine.py:268`), the same 8-thread/5-event
workload lost 23 of 40 writes to `IntegrityError` and the seq assertion went
false. `test_event_seq_is_monotonic_under_concurrent_writers` is load-bearing,
not a test that cannot fail.

- Process signal: Step 8's literal "NOTHING is added to the `scufris` entry"
  contradicts Step 7, which adds the `env.py` import that CREATES that edge. The
  diff resolved it correctly and the close-out records it. The plan was wrong,
  not the work - and R1.1 is the other half of the same plan gap, which no Step
  named at all.
- Process signal: the branch is one squashed implementation commit, so Step 2's
  "write the failing tests first, run each and read the failure" is not
  checkable from history. The close-out names two specific reds and is
  consistent with the code, so it was taken at its word.

No open `manual:` proofs.

## Round 2

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

All ten round-1 findings are ticked. Each Response was re-derived against the
code rather than taken at its word, and one residual remains from R1.2.

- [x] R2.1 (MAJOR) packages/chat/src/scufris_chat/models.py:36 - the R1.2 fix
  closed the actor rule in the NULL domain but not in the value domain the code
  checks. `Actor.__post_init__` (`actors.py:59`) rejects a FALSY `agent_id`,
  while `ck_event_actor_agent_id` requires only `IS NOT NULL`, so
  `actor_kind='agent'` with `actor_agent_id=''` is schema-legal. Reproduced on a
  real migrated database, twice and independently: the INSERT is accepted and
  the next `read_transcript` raises `an agent actor needs an agent id` for the
  WHOLE conversation - the exact failure mode R1.2 was raised to prevent,
  narrowed to the empty string. The reachability argument is R1.2's unchanged,
  and stronger here: a repair INSERT with an uninterpolated variable produces
  `''` more readily than it produces `'smuggled'`. MAJOR rather than MINOR for
  consistency with R1.2, which was the same defect - a rule stated in code as
  truthiness and in the schema as nullability - at the same blast radius. No
  repository convention argues against it: `packages/chat` ships the first
  `CheckConstraint`s in the tree, so there is no column precedent to break.
  Change the constraint to `(actor_kind = 'agent') = (actor_agent_id IS NOT NULL
  AND actor_agent_id <> '')` in `models.py:36` and in revision `18c9104709b8`
  (lines 45-48), and add the `('agent', '')` row to the rejected set in
  `test_migration_creates_the_chat_tables` and `test_actor_must_be_a_known_kind`.
  - Response: fixed in round 3, and WIDER than the finding's literal text. The
    diagnosis is right and its proposed predicate leaves the twin hole:
    `(actor_kind = 'agent') = (actor_agent_id IS NOT NULL AND actor_agent_id <>
    '')` accepts `('operator', '')`, and `_record` (`store.py:176`) builds
    `Actor(ActorKind.OPERATOR, '')`, which `__post_init__` refuses at
    `actors.py:64` with `'' is not None` - the same whole-transcript failure,
    one column value over. The shipped constraint is therefore the full
    predicate `__post_init__` states: `(actor_kind = 'agent' AND actor_agent_id
    IS NOT NULL AND actor_agent_id <> '') OR (actor_kind <> 'agent' AND
    actor_agent_id IS NULL)`, in `models.py:36` and in revision `18c9104709b8`.
    Both empty-string rows - `('agent','')` and `('operator','')` - are asserted
    rejected in `test_migration_creates_the_chat_tables` and
    `test_actor_must_be_a_known_kind`, alongside the accepted `('agent',
    'builder')` that keeps the constraint from passing by rejecting everything.
    Both tests were run RED first and failed on `DID NOT RAISE IntegrityError`,
    not on an import. The docstrings and `README.md:79` now name truthiness, not
    nullability, as the rule - which is the process signal's point, so it is
    written where the next CHECK's author reads it rather than only in the
    review record.

Verification. Full suite green under `nix develop`: `python -m pytest` (exit 0,
no failures), `ruff check .`, `ruff format --check .` (242 files), `mypy .` (242
files), `uv lock --check`, `tatr check`. All six DoD proofs pass on their stated
criteria, including `python examples/chat_conversation.py` (exit 0). The
round-2 diff of `tests/` is additive: no existing test was deleted or weakened.
Revision `18c9104709b8` DDL matches `models.py` column for column including both
CHECKs, and `downgrade` drops in the correct order.

Re-derived independently by the recording pass rather than accepted: a fresh
`upgrade_to_head` database was probed with hand-written INSERTs across the
actor matrix. `('operator','smuggled')`, `('agent',NULL)`, `('system','x')` and
`('wizard',NULL)` are all rejected, `('agent','a1')` and `('operator',NULL)`
accepted - so `ck_event_actor_agent_id` is in the SHIPPED revision and
discriminates both ways rather than rejecting everything. The same probe is what
surfaced R2.1: `('agent','')` is accepted.

- Process signal: R2.1 is the third instance of the shape round 1 already named
  twice - a rule enforced one and a half times. The invariant is written in code
  as truthiness and in the schema as nullability, and the two were assumed to be
  the same predicate. Worth carrying into the lane that adds the next CHECK.

No open `manual:` proofs.

Could not verify: the close-out's exact recorded counts (`1117 passed, 1
skipped`) - this environment's pytest run suppresses the summary line, so exit 0
and the absence of failures stand in for it; and Step 2's TDD ordering, since
the implementation is one squashed commit. Both were taken at their word in
round 1 on the same grounds.

## Round 3

- REVIEWER: out-of-context
- VERDICT: APPROVE

R2.1 is fixed, and the Response's claim to have fixed it WIDER than asked holds.
It is ticked on the out-of-context reviewer's confirmation. Two NITs remain,
both documentation-shaped and neither blocking.

- [ ] R3.1 (NIT) packages/chat/src/scufris_chat/__init__.py:18 - the facade
  docstring still states the actor invariant as it stood before round 2: "a
  CHECK constraint on `actor_kind` is what a hand-written INSERT meets instead".
  There are now TWO check constraints, and `ck_event_actor_agent_id` is the one
  the last two rounds were spent on. `models.py:78`, `EventRow`'s docstring and
  `README.md:79` were all updated to say the rule is truthiness across both
  columns; this file - the one a sibling reads first - was not. Extend the
  bullet to name both CHECKs and the non-empty-vs-NULL predicate.
  - Response:
- [ ] R3.2 (NIT) packages/chat/src/scufris_chat/actors.py:58 -
  `__post_init__`'s docstring promises it refuses a disagreeing actor "however
  it was built", but the branch is `self.kind is ActorKind.AGENT`, an identity
  test. `ActorKind` is a `StrEnum`, so `Actor("agent", None)` with the kind as a
  plain `str` takes the `elif`, sees `agent_id is None` and constructs.
  Re-derived by the recording pass, not taken on the reviewer's word:
  `Actor('agent', None)` builds, and `a.kind` is `str`, so `append_event` then
  dies with `AttributeError` at `store.py:116` instead of the intended
  `ValueError`. Unreachable from `parse` or `_record`, both of which coerce, and
  mypy rejects it repo-wide - so this is a docstring overclaim more than a hole.
  Either coerce with `ActorKind(self.kind)` at the top of `__post_init__`, or
  narrow the docstring to "built with a typed kind".
  - Response:

Verification. Full suite green under `nix develop`, all exit 0: `python -m
pytest` (1 skipped, no failures), `ruff check .`, `ruff format --check .` (242
files), `mypy .` (242 files), `uv lock --check`, `tatr check`, and all six DoD
proofs including `python examples/chat_conversation.py`. The round-3 diff of
`tests/` is additive: no assertion was deleted or weakened, and the new ids and
seqs are distinct, so neither test can pass falsely through a PK or unique
collision.

Re-derived rather than accepted. The out-of-context pass probed a fresh
`upgrade_to_head` database across the actor matrix - four kinds plus `wizard`,
against `None`, `''`, `' '`, `'0'` and `'a1'` - and found the shipped
constraint's accept/reject set exactly equal to `Actor.__post_init__`'s
predicate, with every accepted row round-tripping through `read_transcript`.
Sabotage both ways: reverting the CHECK to round 2's nullability form fails
`test_actor_must_be_a_known_kind` and `test_migration_creates_the_chat_tables`
on `DID NOT RAISE IntegrityError`, and so does the finding's own literal
equality. The recording pass independently confirmed the two facts the whole
round turns on: `Actor(ActorKind.OPERATOR, '')` raises, which is why the
finding's `(actor_kind = 'agent') = (...)` form was insufficient, and
`Actor('agent', None)` constructs, which is R3.2.

- Process signal: the fix being correctly WIDER than the finding is the
  round to learn from. The finding's proposed predicate was accepted as a
  diagnosis and rejected as a patch, and the extra `('operator','')` assertion
  is load-bearing - it goes red against the finding's own text.
- Process signal: R3.1 is the fourth instance of round 1's shape, now in docs
  rather than in code: three of four places that state the actor rule were
  updated and one was missed. The rule is written down in four files; a lane
  that changes it again should expect to grep for all four.

No open `manual:` proofs.

Could not verify: the "run RED first" ordering for Step 2 and for the round-3
fix, since the implementation and each fix are single squashed commits (the
close-out names the two specific reds and is consistent with the code); and the
close-out's exact `1117 passed` count, since this environment suppresses
pytest's summary line - exit 0 and the absence of failures stand in for it, as
in rounds 1 and 2.
