# Delete the legacy JSON import, split the singular agent surface, squash to one baseline revision

- PRIORITY: 101
- TAGS: refactor, v0.2.0, architecture, storage
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260803-213242

## Story

As the Scufris maintainer, I want the unambiguously dead code deleted and the
migration history squashed to one baseline, so that the packages built next
start from a tree with no compatibility layer in it and one revision to reason
about.

This is the SAFE half of the demolition: code with no replacement to wait for.
The agent, session, project and orchestrator stack is NOT touched here - it is
deleted only once the packages that replace it are live.

Planning found that `/api/agent/*` is not what the task assumed. Twelve of its
sixteen routes are the operator console's ONLY door - the settings page, the
tool runner and the session switcher - and the live web pages call them today
with no scheduled repair. Only four routes are true compatibility aliases. So
the surface is SPLIT rather than deleted: the four aliases go, the twelve stay
and move out of a module named `legacy_`. See DECISION.md D1.

## Steps

- [x] Delete the four compatibility aliases from `scufris/api/legacy_agent.py` -
      GET `/api/agent/usage` (`:476`), `/api/agent/memory` (`:484`),
      `/api/agent/account` (`:492`) and `/api/agent/health` (`:328`). Each
      docstring names its `/api/agents/orchestrator/*` twin, which already
      answers out of the same service. Drop the now-unused imports
      (`Capability`, `MemoryFootprint`, `UsageQuota`, `AgentHealth`,
      `AccountInfo` as they fall out).
- [x] Rename the surviving module out of `legacy_`:
      `scufris/api/legacy_agent.py` -> `scufris/api/console.py`,
      `LegacyAgentDeps` -> `ConsoleDeps`, `build_legacy_agent_router` ->
      `build_console_router`, and the `__all__` at `:504`. Update the two import
      sites: `scufris/app.py:60,510` and `tests/conftest.py:196`. Do NOT change
      the `/api/agent/*` URLs (DECISION.md D2).
- [x] Rename `tests/test_legacy_agent_router.py` ->
      `tests/test_console_router.py` and drop only the cases covering the four
      deleted aliases. Everything else is the guard on the console routes and
      stays. `tests/test_app.py:2092`
      (`test_legacy_agent_routes_delegate_to_scoped_diagnostics`) covers the
      delegation of the deleted aliases - retire it and rename what remains.
- [x] Re-point the one live web caller of a deleted alias:
      `web/src/agent-view.ts:153` fetches `/api/agent/usage`; it becomes
      `/api/agents/orchestrator/usage`. Update `web/src/agent-view.test.ts:335,
      349,360` with it. `web/src/agent-settings-view.test.ts:641,670` already
      asserts the settings page does NOT use `/api/agent/health`. DELIVERED
      DIFFERENTLY: that assertion goes trivially true once the route 404s, so
      its subject was moved to `/api/agents/orchestrator/health`, which is the
      route the page could now reach for and must not (REVIEW.md R1.10).
- [x] Drop the four deleted routes from the route-contract table
      (`tests/test_route_contract.py:47,51,55,74` - account, health, memory,
      usage) and re-point `tests/test_release.py:449`, which GETs
      `/api/agent/health`, at `/api/agents/orchestrator/health`.
- [x] Delete `scufris/db/legacy/`, `tests/test_db_legacy.py`,
      `examples/state_migration.py` and `tests/fixtures/legacy_state/`. The JSON
      import path exists only to carry forward data that is being dropped.
      Remove the `LegacyImportRefused` / `import_legacy_state` re-exports and
      the `import_legacy_state` call in `open_state_database`
      (`scufris/db/__init__.py:20,26,30,117`).
- [x] Delete `LegacyImportRow` (`scufris/db/models.py:336-354`) and the
      `legacy_import` table with it.
- [x] Do NOT touch `scufris/config.py`. There is no legacy branch in it: the
      only `legacy` match is `_coerce_legacy_backend` / `canonical_backend`
      (lines 107, 373-401), which folds the `app_server` and `exec` backend ids
      to `codex` and is still load-bearing for persisted agent rows. The same
      holds for `scufris/enums.py`, `scufris/backends/__init__.py`,
      `scufris/agent_store/registry.py`, `web/src/common.ts` and
      `web/src/style.css`, whose `legacy` mentions are all that coercion or an
      unrelated colour alias.
- [x] Delete all five revisions under `scufris/db/migrations/versions/` and
      generate ONE baseline revision by autogenerate against the surviving
      models.
- [x] Refuse a pre-v0.2.0 database with a message that says what actually
      happened (DECISION.md D5). `scufris/db/migrate.py:178` already has
      `_known_revision` and `:211-215` already refuses an unknown revision as
      "written by a newer version" - which a v0.1.x database would now hit, and
      which is the opposite of the truth. Keep a frozenset of the five squashed
      ids (`8f8087f3cc9c`, `9b6587dab793`, `380a27d7fddb`, `3a5161b39846`,
      `e054a39a5fae`) and branch: one of those refuses with a v0.2.0 message
      telling the operator to delete the database; anything else keeps the
      newer-build message, so
      `test_a_database_from_a_newer_scufris_is_refused_without_a_backup`
      (`tests/test_db_migrations.py:374`) keeps its subject. ~10 lines.
- [x] Confirm `alembic upgrade head` builds the schema from empty, and drop
      `legacy_import` from the declared-tables list
      (`tests/test_db_migrations.py:488-495`). Twelve tables survive:
      `projects`, `agents`, `agent_session`, `agent_session_history`,
      `agent_outcome`, `settings_override`, `reasoning_turn`, `auth_session`,
      `schedule`, `digest`, `host_action`, `config_change`.
- [x] Re-point `test_writable_keys_match_the_api_update_model`
      (`tests/test_settings_store.py:189-195`) at
      `scufris.api.console.AgentConfigUpdate`. `api/agents.py`'s `AgentUpdate`
      is NOT an equivalent - it carries per-agent row fields, not the
      `WRITABLE_KEYS` whitelist (DECISION.md D3). The cross-check is kept.
- [x] Delete `tests/test_projects.py:30,326` (the `LegacyImportRefused` import
      and the damaged-`projects.json` assertion). With the import gone nothing
      reads `projects.json`, so the failure it guards cannot occur
      (DECISION.md D4).
- [x] Re-point `examples/host_agent.py:190-192` from
      `scufris.api.legacy_agent.get_backend` to
      `scufris.api.agent_runs.get_backend` - the module its `/api/agents/{id}/
      chat` flow actually goes through (DECISION.md D6). Verify by RUNNING the
      example, not by grep.
- [x] Sweep the stranded references: `tests/test_db_state_boundary.py:39,
      371-482` (the `_legacy_state_dir` fixture at `:374`,
      `test_a_legacy_password_login_still_works_after_the_import` at `:474`, the
      `LegacyImportRow` assertions at `:444-452,466`); `README.md:163-181` (the
      legacy-JSON bullets, the downgrade paragraph and the
      `state_migration.py` pointer); `scufris/app.py:130` and
      `scufris/mcp_server.py:564` (comments describing the legacy import step);
      `tests/test_reasoning_store.py:12` (docstring pointing at the deleted
      `tests/test_db_legacy.py`); `tests/test_agent_run_router.py:8`,
      `scufris/api/agent_runs.py:11` and `scufris/settings_store.py:22`
      (docstring references).
- [x] Update `scufris/README.md`: the module map rows at `:316,473,494`, the
      delegation claim at `:362-364`, the error-table rows at `:563,566,567`,
      the `legacy_import` mention at `:624`, the whole "Reading the legacy JSON
      in - `db/legacy/`" section at `:682-697`, the coercion note at `:720-721`,
      the downgrade pointer at `:739` and the example row at `:759`.
- [x] Note in `CHANGELOG.md` under `[Unreleased]` that the four alias routes and
      the JSON import path are removed, that existing databases are not carried
      forward, and that a leftover `projects.json` is now ignored entirely.

## Definition of Done

- No legacy surface survives
  (cmd: `! rg -q 'legacy_agent|db\.legacy|legacy_import|LegacyImport|state_migration' --glob '!tasks/**' --glob '!CHANGELOG.md' .`).
- The console's twelve routes still answer, and the four aliases 404
  (test: `tests/test_console_router.py`;
  test: `test_the_alias_routes_are_gone`).
- The web console reaches no route the server does not serve: every
  `/api/agent/...` literal under `web/src/` appears in the FastAPI route table
  (test: `test_every_web_api_agent_url_is_served`).
- Exactly one migration revision exists and it builds the schema from empty
  (cmd: `test $(ls scufris/db/migrations/versions/*.py | wc -l) -eq 1`).
- The schema and the revision do not disagree, and no stray table survives the
  squash (test: `test_schema_has_no_pending_autogenerate_diff`;
  test: `test_declared_tables_are_the_only_ones`).
- A pre-v0.2.0 database is refused with a message naming the real cause and the
  fix, and a genuinely newer one still gets the newer-build message
  (test: `test_a_pre_v020_database_is_refused_with_delete_instructions`;
  test: `test_a_database_from_a_newer_scufris_is_refused_without_a_backup`).
- The `WRITABLE_KEYS` cross-check still has a subject
  (test: `test_writable_keys_match_the_api_update_model`).
- The app still starts, serves Stats, and holds a plain orchestrator
  conversation (cmd: `python -m pytest tests/test_app.py -k "stats or chat"`).
- `examples/host_agent.py` runs green against the new patch target
  (cmd: `python examples/host_agent.py`). Blocked on `20260804-041340`: the
  example is broken on master for an unrelated reason (the hostd carve moved
  `tests/test_host_actions.py`), so land that first or the proof cannot be read.
- The gates are green on a tree with no compatibility layer
  (cmd: `nix flake check`).

## Notes

- Parent: 20260803-213242. Decisions: DECISION.md D1-D6. NOTES.md holds the
  original surface inventory; where it and DECISION.md disagree about
  `/api/agent/*`, DECISION.md is current.
- Runs LAST in the carve epic, so the baseline revision is generated against
  models that already live in their final packages. `20260803-214749` (hostctl)
  has landed, so `host_action` / `config_change` are already in place.
- Deleting is the point for the JSON import: no flag, no shim, no deprecation
  window - there is no second consumer and no data worth carrying.
- The console routes are NOT that. They are the operator's only settings page
  and session switcher, and they are deleted when their replacement is live, in
  `20260729-102157`. That epic's children do not currently include the console
  rewrite; if it stays out, the console keeps these routes.
- `nix flake check` will not catch a broken console on its own - the vitest
  suites mock `fetch`. That is what the `test_every_web_api_agent_url_is_served`
  proof is for.
- The rest of the demolition - the agent/session/project/orchestrator stack and
  the pages that render it - belongs to `20260729-102157`.
- Depends on `20260804-041340` (fix the examples the package carve broke), for
  the `examples/host_agent.py` proof only. Everything else is independent.

## Close-out

**What and why.** Three separable demolitions, done as one commit because they
share the doc surface. (1) The four `/api/agent/*` alias routes are gone and the
surviving twelve moved out of a module named `legacy_`
(`api/legacy_agent.py` -> `api/console.py`, `LegacyAgentDeps` -> `ConsoleDeps`,
`build_legacy_agent_router` -> `build_console_router`), URLs unchanged, per D1
and D2. (2) The pre-database JSON import stack is deleted: `db/legacy/`,
`import_legacy_state`, `LegacyImportRefused`, the `legacy_import` table,
`examples/state_migration.py`, `tests/test_db_legacy.py` and the
`tests/fixtures/legacy_state/` tree. (3) The five shipped revisions are squashed
into one autogenerated baseline, `4119562b5fd9_v0_2_0_baseline`, over the twelve
surviving tables.

**Alternatives.** The plan as written deleted the whole singular router. That
premise was overturned during planning and is recorded in DECISION.md; the
alternatives it rejected (delete the router and accept a broken console, delete
the console pages here, rename the URL prefix, narrow the DoD grep) are argued
there rather than repeated here. Within this phase the only open choice was how
to refuse a v0.1.x database: replace the "written by a newer version" message,
or branch ahead of it. Branching won (D5) - it keeps
`test_a_database_from_a_newer_scufris_is_refused_without_a_backup` pointed at a
real case instead of hollowing it out, and `SQUASHED_REVISIONS` is the only
thing left in the tree that can still recognise a v0.1.x database once its
revisions are deleted.

**Difficulties and diagnosis.** D6 named the wrong patch target for
`examples/host_agent.py`. Planning reasoned from the router that SERVES
`/api/agents/{id}/chat` and landed on `scufris.api.agent_runs`; that module does
import `get_backend`, but only for a diagnostics read (`:456`). The call that
LAUNCHES the turn - the one the example's `RecordingBackend` must intercept for
the resumed prompt to be its own recording - is `scufris/orchestrator/runs.py:201`.
Running the example is what caught it; grep would have confirmed the wrong
answer, which is why D6 demanded a run. DECISION.md D6 is amended with the
correction. Second: the example cannot run on this branch for an unrelated
reason - `6d998c8` moved `tests/test_host_actions.py` under `packages/hostd/`
and left both examples' `sys.path` inserts behind. Worked around for the proof
with `PYTHONPATH=packages/hostd/tests`; the real fix is `20260804-041340`.

**Evidence.**

- `rg 'legacy_agent|db\.legacy|legacy_import|LegacyImport|state_migration'`
  outside `tasks/` and `CHANGELOG.md`: no hits.
- `ls scufris/db/migrations/versions/*.py | wc -l` -> 1.
- Full pytest green, including `test_the_alias_routes_are_gone`,
  `test_every_web_api_agent_url_is_served`,
  `test_a_pre_v020_database_is_refused_with_delete_instructions`,
  `test_a_database_from_a_newer_scufris_is_refused_without_a_backup`,
  `test_schema_has_no_pending_autogenerate_diff`,
  `test_declared_tables_are_the_only_ones` and
  `test_writable_keys_match_the_api_update_model`.
- `npm test` in `web/`: 262 passed, 25 files.
- `examples/host_agent.py` with `PYTHONPATH=packages/hostd/tests`: exit 0,
  through the denial and the resumed prompt. The DoD proof as written stays
  BLOCKED on `20260804-041340` landing; it is the only unproven line.
- `nix flake check` green.

**Round 1 fixes.** Ten findings, all fixed; responses on each in REVIEW.md. The
one with teeth was R1.1: the squash removed the only "behind head" state the
tree could build, and the first attempt retired
`test_the_backup_is_taken_on_the_real_migration_path` behind a `pytest.skip`
rather than re-pinning it - which left `upgrade_to_head`'s copy-before-migrate
step deletable with the suite green. The `behind_head` fixture stages a tmp
Alembic environment with a throwaway follow-on revision instead (DECISION.md
D7); sabotage-checked. R1.2-R1.4 were three wrong claims in
`scufris/README.md`'s prose about `AgentDiagnostics`, all of them survivors of a
by-symbol sweep. R1.6 replaced a false comment with a real comment-stripper in
the web-URL extractor. R1.5's process signal is acted on rather than noted:
`flake.nix` now runs `ruff format --check .` as its own check, because the gate
running only `ruff check` is how an over-long line reached review.

**Reflection.** The doc sweep was larger than the code change and is where the
risk sat: `scufris/README.md` described the import in a section of its own, an
error-table row, a module-map row and two cross-references, and the root README
built its whole backup-and-downgrade story on legacy files still existing. A
grep for `legacy` finds those; a grep for the deleted symbols does not, which is
why the DoD grep passing is necessary and not sufficient. Next time, sweep the
prose by CONCEPT before sweeping by symbol. The other lesson is D6's: a planned
patch target is a hypothesis about which module is on the hot path, and only
running the thing settles it.
