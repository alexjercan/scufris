# Assemble provider context from the semantic conversation

- PRIORITY: 98
- TAGS: feature, v0.2.0, lane1, chat
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
- PARENT: 20260801-154211
- DEPENDS ON: 20260804-115256

## Story

As the operator, I want the conversation to survive `/new`, compaction, a
backend switch and a restart, so that the provider session is a cache I can
throw away rather than the place my history actually lives.

Without this the release's headline promise is unimplementable. Nothing in the
sprint plan covered it before the 2026-08-04 lane cut.

## Steps

- [ ] Write `tasks/20260804-115320/DECISION.md` FIRST, before the code, because
      three of its answers decide the shape below. Sections:
      1. **The cache row is keyed for LOOKUP by the triple, and PRIMARY-KEYED by
         `(conversation_id, backend)`.** `policy_version` is a constrained
         column that the read must match, not a third key column. With the
         triple as the primary key a policy DOWNGRADE would resurrect a binding
         seeded under the old rules that has since missed every event appended
         under the new one - a stale session read as warm. One live binding per
         `(conversation, backend)`, re-seed OVERWRITES it, and nothing
         accumulates.
      2. **SUMMARIZATION DEFERRAL.** v0.2.0 bounds assembled context with a
         WINDOW, not a summarizer; `NOTES.md` holds the four-item evidence.
         Write it as a deferral with its reopening trigger - the first time a
         window drops context the operator actually needed - not as a gap.
         Summary versioning falls away with it: there is no summary to version,
         and `policy_version` survives for what `NOTES.md` says it is for (the
         system/project policy and the presets legal right now).
      3. **EAGER VERSUS LAZY RE-SEED: LAZY.** A restart and a provider-side
         compaction invalidate a binding with NO switch event to hang eager work
         on, so the lazy path - miss at the next turn, assemble, re-seed - has to
         exist whatever is decided here. Eager would be a SECOND path doing the
         same thing earlier: a mode with no requirement in v0.2.0. So a backend
         switch writes nothing at all; it is "use backend B next turn", and the
         cache miss does the rest. Record eager as rejected with its reopening
         trigger: a measurable, operator-visible stall on the first turn after a
         switch.
      4. **The bound is an EVENT COUNT, and that is a proxy.** One enormous body
         still overflows a provider that a hundred small ones would not. A
         character or token bound is deferred with its own trigger: the first
         time a windowed assembly still overflows a provider. Two knobs before
         either has a real caller would be one knob too many.
- [ ] Write `packages/chat/tests/test_chat_sessions.py` FIRST, red, over the
      `database` fixture pattern `test_chat_events.py` and
      `test_chat_delivery.py` already use (file-backed `open_database`, tables
      from `Base.metadata`, `OWNED_TABLES` grown to `("conversation", "event",
      "delivery", "provider_session")`). Five tests, named in Definition of Done.
      `pytest packages/chat/tests -q -k session` currently exits 5, no tests
      collected - that is the red this step turns green.
- [ ] Add `ProviderSessionRow` to `packages/chat/src/scufris_chat/models.py`:
      `conversation_id` and `backend` as a composite `PrimaryKeyConstraint`,
      plus `policy_version`, `provider_session_id` and `seeded_at`. No FOREIGN
      KEYs, for the reason that module's docstring already records. Three CHECKs,
      following `_DELIVERY_CHANNEL_CHECK`'s truthiness reasoning - the database
      is what a repair session meets, not the store:
      - `backend <> ''` - it is half the key, so an empty one files the binding
        under a backend nothing looks up while the real one re-seeds forever.
      - `provider_session_id <> ''` - a binding to no session reads as a warm
        cache and then resumes nothing.
      - `policy_version >= 1` - the column the whole invalidation rests on. A
        zero or negative version is not an older policy, it is a corrupt row.
- [ ] Add to `packages/chat/src/scufris_chat/store.py`, each taking the caller's
      OPEN `Connection` first and `conversation_id` second, the order R1.11 of
      `tasks/20260804-115319/REVIEW.md` aligned the whole surface on:
      - `CONTEXT_POLICY_VERSION: int` and `CONTEXT_WINDOW_EVENTS: int` - the
        current assembly policy and the window. The version is a REQUIRED keyword
        on both cache functions rather than baked in, because a key component the
        caller cannot see is one it cannot reason about, and a re-seed has to be
        legible at the call site.
      - `cached_session(conn, conversation_id, *, backend, policy_version) ->
        SessionBinding | None` - `None` for an absent row, a row under a
        different `policy_version`, AND an unknown conversation. A miss is
        NORMAL; nothing here raises. This is the one function on the surface that
        does not refuse an id it cannot resolve, and the docstring says why.
      - `bind_session(conn, conversation_id, *, backend, policy_version,
        provider_session_id) -> SessionBinding` - UPSERT via
        `sqlite_insert(...).on_conflict_do_update(...)`, the form
        `claim_delivery` and `scufris/scheduler.py:133` already use, so a
        re-seed replaces the stale binding rather than colliding with it. It
        DOES refuse an unknown `conversation_id` with a `LookupError`, the same
        check `append_event` makes; factor that check out as
        `_require_conversation` and call it from both rather than duplicating it.
      - `assemble_context(conn, conversation_id, *, max_events=
        CONTEXT_WINDOW_EVENTS) -> str` - the seed prompt for a re-seed.
      - `SessionBinding`, a frozen record like `EventRecord`.
- [ ] Bound `assemble_context` IN SQL, not by slicing a full read.
      `ORDER BY event_seq DESC LIMIT max_events`, then reverse - a private
      `_recent_events`. `format_fork_seed`'s `kept = context[-max_turns:]`
      (`scufris/sessions/transcript.py:177`) is what this generalizes, and the
      generalization is exactly this: it slices AFTER loading the whole
      conversation, which is unbounded work to produce a bounded result. The
      existing `UniqueConstraint(conversation_id, event_seq)` is the index that
      query uses; no new index.
- [ ] Render every line of the assembled context ATTRIBUTED to its actor -
      `operator`, `agent:<id>`, `orchestrator`, `system` - and say in the
      preamble that only the operator's lines are instructions. Per
      `tasks/20260729-220835/DECISION.md` section 3, an agent report enters
      context as an attributed, untrusted quotation. `20260804-115321` depends on
      this task partly to assert the negative
      (`test_assembled_context_does_not_relabel_agent_as_operator`); the format
      it asserts against is built here.
- [ ] Export `SessionBinding`, `assemble_context`, `bind_session`,
      `cached_session`, `CONTEXT_POLICY_VERSION` and `CONTEXT_WINDOW_EVENTS`
      from `packages/chat/src/scufris_chat/__init__.py` and `__all__`.
      `ProviderSessionRow` stays private, as `EventRow` and `DeliveryRow` are.
      Correct the module docstring's "Three tables and seven functions" to four
      and ten.
- [ ] Generate the Alembic revision with `down_revision = "53aaa107ce2d"` (the
      current head - `53aaa107ce2d_chat_delivery.py`) by AUTOGENERATE, not by
      hand, and confirm `test_schema_has_no_pending_autogenerate_diff` is green -
      that test is what proves the revision matches the models.
- [ ] Grow `tests/test_db_schema.py`: add `"provider_session"` to
      `test_declared_tables_are_the_only_ones` and its docstring, and add
      `test_migration_creates_the_provider_session_table` asserting the composite
      primary key and all three CHECKs by INSERTing against them, the way
      `test_migration_creates_the_delivery_table` does - a constraint SQLite
      parsed but does not enforce would still appear in `sqlite_master`.
      Check the file against the 900-line test cap first
      (`python scripts/check_file_size.py`); its ALLOWLIST is a ratchet no entry
      may be added to, and splitting `test_db_migrations.py` is how
      `20260804-115319` handled the same pressure.
- [ ] Grow `examples/chat_conversation.py` with a step 6: bind a session on
      backend `codex` and show the cache HIT, then switch to `claude` and show
      the MISS, print the assembled context, bind the new provider id, and
      re-print the transcript. Assert the transcript is byte-identical across the
      switch and that the two provider session ids differ, with a non-zero exit
      if either does not hold. Keep the script offline and `scufris`-free; it is
      gated by `tests/test_examples.py`. The epic's Lane 1 blurb also wants a
      colour-per-actor rich transcript - that is NOT in this task's steps and
      stays out; note it for Lane 8.
- [ ] Add a section 6 to `packages/chat/src/scufris_chat/README.md` for the
      cache, the window and the two deferrals, renumbering "The surface" to 7 and
      "What is not here yet" to 8. Correct section 1's "three tables" table and
      the `conversation` note that currently says the cache "lives with the
      cache" without saying where that is. Update the surface table with the
      three new functions. Link `DECISION.md`.
- [ ] Update the one-line module maps that name this package's contents:
      `AGENTS.md`'s `packages/chat` row and `scufris/README.md:480`. Add a
      CHANGELOG.md entry, as `20260804-115319` did.
- [ ] Run `nix develop -c python -m pytest packages/chat/tests
      tests/test_db_migrations.py tests/test_db_schema.py tests/test_examples.py
      tests/test_package_boundaries.py -q`, then `nix flake check` and
      `tatr check`.

## Definition of Done

- Switching backend re-seeds from the conversation and preserves the semantic
  transcript exactly: the lookup under the new backend misses, the transcript
  read before and after the switch is equal, and the two provider session ids
  differ (test: `test_backend_switch_preserves_the_conversation`).
- Assembled context stays within its configured bound for a conversation far
  larger than the bound - the newest `CONTEXT_WINDOW_EVENTS` bodies are present
  and the oldest are not (test: `test_assembled_context_is_bounded`).
- A session bound under an older policy version is not reused: the lookup
  returns `None` rather than raising (test:
  `test_stale_policy_version_forces_reseed`).
- A valid cached session is reused rather than re-seeded - the lookup returns
  the same `provider_session_id` that was bound
  (test: `test_valid_session_is_not_reseeded`).
- Assembled context attributes every line to its actor, so an agent report is a
  quotation rather than something the operator said
  (test: `test_assembled_context_attributes_its_actors`).
- The shipped migration builds the table with its composite key and all three
  CHECKs, asserted by INSERTing against them (test:
  `tests/test_db_schema.py::test_migration_creates_the_provider_session_table`).
- `provider_session` is the only table the schema gained
  (test: `tests/test_db_schema.py::test_declared_tables_are_the_only_ones`).
- `examples/chat_conversation.py` exits 0 and prints a backend switch whose
  transcript is unchanged and whose provider session id is not
  (test: `tests/test_examples.py`).

  cmd: `nix develop -c python -m pytest packages/chat/tests
  tests/test_db_schema.py tests/test_examples.py -q`
  (red on base: `nix develop -c python -m pytest packages/chat/tests -q -k
  session` exits 5, no tests collected - verified 2026-08-04)

## Notes

- Source: `tasks/20260729-220835/DECISION.md` section 1 and its Consequences.
- Two of the six questions the spike deferred "to v0.3.0 tasks" are answered
  here rather than in a container that no longer exists.
- Lane 1 of `tasks/20260801-154211/TASK.md`.
- The table is named `provider_session`, not `session`: `agent_session`,
  `agent_session_history` and `auth_session` already exist, and the word this
  row means is the PROVIDER's.
- No invalidation function, deferred with reason. `/new` mints a new
  `conversation_id` (`tasks/20260729-220835/DECISION.md` section 6), so the old
  binding is not dropped - it is simply never looked up again. A
  `forget_session` would be a function with no caller.
- No `seeded_through_seq` column, deferred with reason. Events appended to a
  conversation while a binding is warm - an agent report landing between turns -
  are not in the provider's working memory, and nothing in v0.2.0 detects that.
  It is the turn-driving code's problem, and that code does not exist in this
  lane. Reopening trigger: the first caller that appends a non-operator event
  outside a turn it is itself driving.
- `20260804-115321` depends on this task as well as on `20260804-115256`,
  because the assembled-context path its
  `test_assembled_context_does_not_relabel_agent_as_operator` guards is built
  here.
- `packages/chat` depends on `scufris_core` alone, and
  `tests/test_package_boundaries.py` is what keeps that true. Nothing in these
  steps adds a dependency - assembly reads rows and returns a string; no
  backend, no network, no model.
