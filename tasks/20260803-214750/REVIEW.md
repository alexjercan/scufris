# Review: Delete the legacy JSON import, split the singular agent surface, squash to one baseline revision

- TASK: 20260803-214750
- BRANCH: refactor/split-agent-surface-squash-baseline

## Round 1

- REVIEWER: out-of-context (lanes: behavior/proofs, correctness/security/persistence, design/standards/docs)
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) tests/test_db_migrations.py:66 - `_previous_revision` now
  calls `pytest.skip` when head has no parent, which is unconditional on this
  branch: head IS the baseline. So
  `test_the_backup_is_taken_on_the_real_migration_path` (`:274`) never executes.
  It was the ONLY test reaching `backup_database` through `upgrade_to_head`
  (`scufris/db/migrate.py:241`); the other two (`:260`, `:339`) call
  `backup_database` directly. The squash changed how the
  backup-before-migrate invariant is enforced and the assertion was retired
  rather than re-pinned - deleting `migrate.py:241` now leaves the suite green.
  Fix: build the behind-head state without shipping a second revision -
  `monkeypatch` the migrations dir at a tmp copy of `versions/` holding the
  baseline plus a throwaway follow-on revision, then assert the `.bak` as
  before. If that is judged out of scope, record the lost proof as a
  DECISION.md consequence so the next revision restores it deliberately.
  Response: fixed in this commit. `_previous_revision` and its skip are gone;
  a `behind_head` fixture (`tests/test_db_migrations.py:62-127`) copies the
  shipped Alembic environment to a tmp dir, writes one throwaway follow-on
  revision into the copy and monkeypatches `migrate._migrations_dir` at it, so
  `head_revision`, `_alembic_config` and `upgrade_to_head` all see a head one
  step past the baseline while the production path runs unmodified. The
  post-migration discriminator moves from `config_change` to the throwaway's
  `throwaway_probe`. Confirmed by sabotage: deleting `migrate.py:241` now fails
  `test_the_backup_is_taken_on_the_real_migration_path`. `pytest -rs` on the
  module reports 22 passed, 0 skipped. Recorded as DECISION.md D7 with the two
  rejected alternatives.

- [x] R1.2 (MAJOR) scufris/README.md:316 - the `api/console.py` module-map row
  still lists `AgentDiagnostics` in `ConsoleDeps`, but this diff removed that
  field (`scufris/api/console.py:162-168` is settings/agents/store/runs/
  supervisor/api_token, and `scufris/app.py:510-517` passes no `diagnostics=`).
  The Step at TASK.md:120-122 names `:316` and is ticked, so this is an
  undelivered clause, not just drift. Drop `AgentDiagnostics` from that cell.
  Response: fixed in this commit. `AgentDiagnostics` dropped from the
  `ConsoleDeps` cell (`scufris/README.md:316`).

- [x] R1.3 (MAJOR) scufris/README.md:360-364 - "The singular `/api/agent/*`
  family and the scoped `/api/agents/{id}/*` family are handed the SAME
  `AgentRunService` and `AgentDiagnostics` instances" is false after the alias
  deletion: the console router is handed no `AgentDiagnostics` at all. The
  cited proof, `test_the_console_router_reaches_for_nothing`
  (`tests/test_console_router.py:467`), booby-traps four constructors and
  asserts the router builds nothing ambient - it does not pin shared instances.
  The Step at TASK.md:120-122 names `:362-364` and is ticked. Restate the
  sentence as `AgentRunService` only, and either cite a test that actually pins
  sharing or drop the citation.
  Response: fixed in this commit. The sentence now says the two families share
  the same `AgentRunService` and that the console router is handed no
  `AgentDiagnostics` at all, and the citation is restated as what
  `test_the_console_router_reaches_for_nothing` actually pins - that the router
  builds nothing ambient of its own - rather than as a sharing proof
  (`scufris/README.md:360-366`).

- [x] R1.4 (MAJOR) scufris/README.md:416-419 - "The console's singular
  `/api/agent/*` family reads through the same service - `info` and `config`
  resolve `_require_agent(ORCHESTRATOR_ID)` and delegate to it, envelopes
  included" is wrong on three counts: `info`/`config` call `require_agent`
  re-exported from `api/agent_runs.py`, not `AgentDiagnostics`; the helper is no
  longer named `_require_agent`; and no surviving console route returns a
  `Capability` envelope. Replace with "no console route consumes
  `AgentDiagnostics`; the envelopes live only on `/api/agents/{id}/*`."
  Response: fixed in this commit. Replaced with "No console route consumes
  `AgentDiagnostics`; the envelopes live only on `/api/agents/{id}/*`", followed
  by what the console DOES share: `require_agent(deps.runs, ORCHESTRATOR_ID)`
  re-exported from `api/agent_runs.py`, with the model and auth mode read off
  the record (`scufris/README.md:414-421`).

- [x] R1.5 (MINOR) tests/test_app.py:857 - `ruff format --check .` reports
  `tests/test_app.py` and `tests/test_console_router.py` would both be
  reformatted (line length 88, pinned in AGENTS.md). It passes the gate only
  because `flake.nix:251` runs `ruff check .` and never `ruff format --check`.
  Run `ruff format` on both files.
  Response: fixed in this commit. `ruff format` run on both files; the tree is
  format-clean (233 files, 0 would reformat). Also acted on the process signal:
  `flake.nix` gains a `ruff-format` check running `ruff format --check .`, so
  the gate can no longer miss this, and the two places that enumerate the gate
  (`README.md:405`, `scufris/README.md:707`) are updated with it.

- [x] R1.6 (MINOR) tests/test_route_contract.py:254-256 - the comment claims
  anchoring on a quote stops a `/api/agent/*` in a COMMENT being read as a URL,
  but backticks are in the character class, so backticked routes inside `//`
  comments DO match: `web/src/agent-settings-view.ts:74-75,81` contributes
  `/api/agent/mcp` and two `/api/agents/{id}/*` paths to the reached set. The
  gate passes only because those happen to be served; a comment naming a retired
  route would fail it spuriously. Strip `//` lines and `/* */` blocks before
  matching, or delete the false claim from the comment.
  Response: fixed in this commit. `_without_comments`
  (`tests/test_route_contract.py:262-305`) blanks `//` lines and `/* */` blocks
  before the match, written as a scan rather than a regex so a `//` inside a
  string literal (`https://...`) is not mistaken for a comment. The false claim
  is gone from the `WEB_API_AGENT_URL` comment, which now says the quote anchor
  does NOT separate URLs from prose and points at the stripper.
  `test_a_route_named_in_a_comment_is_not_a_reached_url` pins both comment forms
  and the `https://` case.

- [x] R1.7 (MINOR) tests/test_db_migrations.py:244-245 - the docstring of
  `test_the_backup_is_a_whole_readable_database` points the wiring proof at
  `test_the_backup_is_taken_on_the_real_migration_path`, which now always skips.
  Fix with R1.1, or say plainly that the wiring is unproven until a second
  revision lands.
  Response: fixed by R1.1. The docstring's pointer is accurate again -
  `test_the_backup_is_taken_on_the_real_migration_path` executes and proves the
  wiring. No wording change needed.

- [x] R1.8 (NIT) scufris/db/migrate.py:232 - the pre-v0.2.0 refusal deliberately
  writes no `.bak` (pinned at tests/test_db_migrations.py:434) yet tells the
  operator to "Delete the database and start again". Add "keep a copy first if
  you want the old rows" so a v0.1.x operator is not told to destroy their only
  copy.
  Response: fixed in this commit. The refusal now reads "Delete the database and
  start again - keep a copy first if you want the old rows"
  (`scufris/db/migrate.py:232-236`).

- [x] R1.9 (NIT) README.md:412-415 - the `/api/agents/orchestrator/health`
  substitution left a ragged wrap ("...and the settings / view - so you / can
  tell what is deployed"). Rewrap the paragraph.
  Response: fixed in this commit. Paragraph rewrapped (`README.md:412-414`).

- [x] R1.10 (NIT) web/src/agent-settings-view.test.ts:670 - the Step at
  TASK.md:79-82 says this assertion "stays true and its parenthetical is now
  stale wording only", but the diff changed its subject from
  `/api/agent/health` to `/api/agents/orchestrator/health`. The substitution is
  defensible - the old form goes trivially true once the route 404s - but the
  tick does not describe it. Note the change in the Step text or the close-out.
  Response: fixed in this commit. The Step at TASK.md:79-83 now records the
  substitution explicitly - the old assertion goes trivially true once the route
  404s, so its subject moved to `/api/agents/orchestrator/health`, which is the
  route the settings page could now reach for and must not.

- Process signal: `nix flake check` is green and the close-out says so
  accurately, but the gate never runs `ruff format --check`, which is how R1.5
  reached review. Worth deciding whether the flake should add it.
- Process signal: R1.2-R1.4 are all concept-level claims in `scufris/README.md`
  (dependency graph, delegation, envelopes) that both the DoD grep and a
  symbol-level sweep pass over cleanly. The doc sweep needs a by-concept pass,
  not only a by-symbol one.
- Process signal: the behavior/proofs lane returned after the round was first
  committed; R1.10 and this note are its amendment, made before any Response
  was written. It independently reached R1.1-R1.4 at MINOR where the other two
  lanes and the recording pass put them higher.

Pending, not blocking: the DoD proof `python examples/host_agent.py` cannot be
read on this branch. It fails at `examples/host_agent.py:46`
(`ModuleNotFoundError: No module named 'test_host_actions'`) from the `sys.path`
insert at `:43`, which is byte-identical on master and which this diff does not
touch. The cause is `6d998c8` moving `tests/test_host_actions.py` into
`packages/hostd/tests/`, already filed as `20260804-041340`. The TASK.md DoD and
DECISION.md D6 both record the block honestly. This is a dependency, not a
finding against this diff. The behavior lane reproduced D6's own workaround -
`PYTHONPATH=packages/hostd/tests python examples/host_agent.py` exits 0 and runs
through the denial and the resumed prompt - so the re-point to
`scufris.orchestrator.runs` is confirmed correct and the close-out's account of
it is honest. It stays unproven BY THE GATES until `20260804-041340` lands.

Verified by the recording pass: `nix flake check` green (ruff, mypy, full
pytest, tatr check, filesize). DoD proof 1 (the `legacy_*` grep) clean outside
`tasks/` and `CHANGELOG.md`; proof 4 (one revision file) holds.
`scufris/api/legacy_agent.py` is gone and `console.py` carries the rename with
its docstring rewritten and the five alias-only imports pruned.
`web/src/agent-view.ts:154` is re-pointed to
`/api/agents/orchestrator/usage`, and `test_writable_keys_match_the_api_update_model`
re-points at `scufris.api.console.AgentConfigUpdate` per D3. The four deleted
tests all cover the removed import path itself, not behavior that survives; D4
records the `projects.json` consequence. `pytest -rs` on
`tests/test_db_migrations.py` shows exactly one skip, which is R1.1. The
correctness lane checked the new baseline op-by-op against the five deleted
revisions and found the twelve surviving tables, columns, types, nullability and
both `seq` unique constraints reproduced, with `legacy_import` the only
deliberate drop.

## Round 2

- REVIEWER: out-of-context (lanes: behavior/proofs+tests, design/standards/docs+honesty)
- VERDICT: APPROVE

All ten round-1 findings are confirmed fixed and ticked above. Nothing in the
fix commit `3bebed6` regressed. What remains is four record-and-prose defects
the round-1 rewrites introduced or left behind, none of them blocking.

- [ ] R2.1 (MINOR) tasks/20260803-214750/TASK.md:107 - the ticked Step reads
  "Re-point `examples/host_agent.py:190-192` from
  `scufris.api.legacy_agent.get_backend` to `scufris.api.agent_runs.get_backend`",
  but what landed patches `scufris.orchestrator.runs`
  (`examples/host_agent.py:190-195`). D6's amendment and the close-out both
  record the correction; the Step text does not. R1.10 established the remedy
  for exactly this shape and it was applied to only one of the two stale ticks.
  Append a `DELIVERED DIFFERENTLY:` clause naming `scufris.orchestrator.runs`
  and pointing at DECISION.md D6.
  - Response:

- [ ] R2.2 (NIT) scufris/README.md:365 - the R1.3/R1.4 rewrites left ragged
  wraps of the same class R1.9 was raised for: line 365 is 43 chars, 419 is 61
  and 423 is 65, each mid-paragraph between 76-80 char neighbours. Rewrap both
  paragraphs (`:360-368`, `:414-425`) to the file's ~80-col fill.
  - Response:

- [ ] R2.3 (NIT) flake.nix:253 - the new comment justifies the check by citing
  "20260803-214750 REVIEW.md R1.5". `AGENTS.md:107` is explicit: "Task IDs
  belong in task records and Markdown, never in code comments or docstrings".
  Restate it as a fact about the code, e.g. "`ruff check` cannot catch this:
  the lint selection has no `E501`, so `line-length` is enforced only by the
  formatter."
  - Response:

- [ ] R2.4 (NIT) scufris/README.md:707 - the proof-table row was edited for
  `ruff format --check` but still omits the `filesize` check that
  `flake.nix:271` runs. `README.md:405`, edited in the same commit, does list
  it. Add the file-size guard so the two enumerations agree with `flake.nix`.
  - Response:

- Process signal: the round-1 response acted on R1.5's process signal by adding
  a repository-wide gate (`flake.nix` `ruff-format`) rather than only noting it.
  The change is small and justified, but it is recorded only in REVIEW.md and
  the close-out - a new gate on every future diff is the kind of choice
  DECISION.md otherwise carries.
- Process signal: both lanes independently reached the ragged-wrap finding, and
  it is the second round in a row that a prose rewrite left one behind. The doc
  sweep owes a reflow pass after any paragraph edit, not just a content check.

Verified by the recording pass. Both lanes re-derived every round-1 Response
independently; each is CONFIRMED with evidence, and each of the four new
findings above was reproduced by me before recording. R1.1 was sabotage-checked
two ways: deleting the `backup_database` call and its log line at
`scufris/db/migrate.py:242-243` fails exactly
`test_the_backup_is_taken_on_the_real_migration_path` with
`no such table: projects`, and stubbing `backup_database` to a no-op fails the
same test through `migrate.py:242`. `pytest tests/test_db_migrations.py -rs`
reports 22 passed and 0 skipped, so the skip R1.1 was filed for is gone.
`_without_comments` was probed directly against `//` inside a string literal,
`https://`, block comments, a nested `/* a /* b */`, an apostrophe inside a `//`
comment and a regex character class; its only mis-parse mode under-strips, so it
cannot hide a real fetch URL. `nix flake check` runs 8 checks, all passed, exit
0, with `ruff-format` among them. I re-derived DoD proof 1 (the `legacy_*` grep,
no hits outside `tasks/` and `CHANGELOG.md`) and proof 4 (one revision file), and
read the `behind_head` fixture and the console/`AgentDiagnostics` prose against
`scufris/api/console.py` and `scufris/app.py` myself. `pytest tests/test_app.py
-k "stats or chat"` -> 19 passed.

Pending, not blocking, unchanged from round 1: DoD proof 9
(`python examples/host_agent.py`) cannot be read on this branch. It fails at
`examples/host_agent.py:46` from a `sys.path` insert that is byte-identical on
master and that this diff does not touch, caused by `6d998c8` moving
`tests/test_host_actions.py` into `packages/hostd/tests/` and already filed as
`20260804-041340`. Both lanes reproduced D6's workaround -
`PYTHONPATH=packages/hostd/tests python examples/host_agent.py` exits 0 and runs
through the denial and the resumed prompt - so the re-point is confirmed
correct. It stays unproven BY THE GATES until `20260804-041340` lands. There are
no `manual:` proofs on this task.
