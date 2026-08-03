# Delete the legacy agent router and JSON import, squash to one baseline revision

- PRIORITY: 101
- TAGS: refactor, v0.2.0, architecture, storage
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -
- PARENT: 20260803-213242

## Story

As the Scufris maintainer, I want the unambiguously dead code deleted and the
migration history squashed to one baseline, so that the packages built next
start from a tree with no compatibility layer in it and one revision to reason
about.

This is the SAFE half of the demolition: code with no replacement to wait for.
The agent, session, project and orchestrator stack is NOT touched here - it is
deleted only once the packages that replace it are live.

## Steps

- [ ] Delete `scufris/api/legacy_agent.py` and `tests/test_legacy_agent_router.py`.
      The `/api/agent/*` surface exists only for backwards compatibility, which
      is no longer a goal.
- [ ] Delete `scufris/db/legacy/`, `tests/test_db_legacy.py` and
      `examples/state_migration.py`. The JSON import path exists only to carry
      forward data that is being dropped.
- [ ] Delete the `legacy_import` table and its revision.
- [ ] Remove the legacy router from `scufris/app.py`. Do NOT touch
      `scufris/config.py`: there is no legacy branch in it. The only `legacy`
      match is `_coerce_legacy_backend` / `canonical_backend` (lines 107,
      373-401), which folds the `app_server` and `exec` backend ids to `codex`
      and is still load-bearing for persisted agent rows.
- [ ] Delete all five revisions under `scufris/db/migrations/versions/` and
      generate ONE baseline revision by autogenerate against the surviving
      models.
- [ ] Refuse a pre-v0.2.0 database with a message that says what actually
      happened. Its `alembic_version` holds `e054a39a5fae`, which the new
      baseline does not know, so `upgrade_to_head` raises "Can't locate
      revision" - and the existing unknown-revision path
      (`tests/test_db_migrations.py:374`) tells the operator their database
      came from a NEWER scufris, which is the opposite of the truth. Detect an
      unknown revision and refuse with a v0.2.0-specific message telling the
      operator to delete the database. ~10 lines in `migrate.py`.
- [ ] Confirm `alembic upgrade head` builds the schema from empty, and that the
      pending-autogenerate-diff test is green against the new baseline.
- [ ] Sweep the references the deletion strands, which are wider than the two
      lines an early estimate suggested:
      `tests/test_db_state_boundary.py:371-482` (fixture `_legacy_state_dir` at
      `:374`, `test_a_legacy_password_login_still_works_after_the_import` at
      `:474`, assertions at `:444-452`); `README.md:163-177` plus the
      `state_migration.py` pointer at `:181`; `scufris/app.py:132` and
      `scufris/mcp_server.py:563` (comments describing the legacy import step);
      `tests/test_reasoning_store.py:12` (docstring pointing at the deleted
      `tests/test_db_legacy.py`).
- [ ] Re-point or retire `test_writable_keys_match_the_api_update_model`
      (`tests/test_settings_store.py:192`). It asserts
      `AgentConfigUpdate.model_fields == WRITABLE_KEYS`, and `AgentConfigUpdate`
      is defined ONLY in `api/legacy_agent.py:93` - so deleting the router
      deletes the guard on a hand-kept whitelist. If `api/agents.py` has an
      equivalent update model, re-point at it; if not, record the lost
      cross-check as a consequence rather than losing it silently.
- [ ] Check `tests/test_projects.py:326`, which expects `LegacyImportRefused`
      from a project-store path. Confirm the underlying behavior - a damaged
      `projects.json` beside a database - has no remaining meaning rather than
      assuming it.
- [ ] Note in `CHANGELOG.md` that the legacy agent API and the JSON import path
      are removed and that existing databases are not carried forward.

## Definition of Done

- No legacy surface survives
  (cmd: `! rg -q 'legacy_agent|db\.legacy|legacy_import' --glob '!tasks/**' --glob '!CHANGELOG.md' .`).
- Exactly one migration revision exists and it builds the schema from empty
  (cmd: `test $(ls scufris/db/migrations/versions/*.py | wc -l) -eq 1`).
- The schema and the revision do not disagree, and no stray table survives the
  squash (test: `test_schema_has_no_pending_autogenerate_diff`;
  test: `test_declared_tables_are_the_only_ones`).
- A pre-v0.2.0 database is refused with a message naming the real cause and the
  fix, not the "newer scufris" message
  (test: `test_a_pre_v020_database_is_refused_with_delete_instructions`).
- The app still starts, serves Stats, and holds a plain orchestrator
  conversation (cmd: `python -m pytest tests/test_app.py -k "stats or chat"`).
- The gates are green on a tree with no compatibility layer
  (cmd: `nix flake check`).

## Notes

- Parent: 20260803-213242.
- Runs LAST in the carve epic, so the baseline revision is generated against
  models that already live in their final packages.
- Deleting is the point. Do not preserve any of this behind a flag, a shim or a
  deprecation window - there is no second consumer and no data worth carrying.
- The rest of the demolition - the agent/session/project/orchestrator stack and
  the pages that render it - belongs to 20260729-102157, after its replacement
  is live.
