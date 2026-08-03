# Notes: Delete the legacy agent router and JSON import, squash to one baseline revision

Goal in one line: remove the two compatibility surfaces that have no replacement
to wait for - `/api/agent/*` and the pre-database JSON import - and collapse five
Alembic revisions into one baseline generated against the models as they stand
after the carve.

~2600 lines deleted, 5 revisions replaced by 1, and startup loses a step.

## What changes

What an OPERATOR sees:

| Before | After |
|---|---|
| `/api/agent/*` answers (aliases of `/api/agents/orchestrator/*`) | 404 |
| an existing state dir with `projects.json` etc. is imported on first start | ignored |
| an existing `scufris.db` upgrades to head | **it does not** - see open question 1 |
| `python examples/state_migration.py` documents the upgrade | the example is gone |

What a MAINTAINER sees: five revisions become one, `open_state_database` is
open -> upgrade instead of open -> upgrade -> import, and no module in the tree
carries the word `legacy` except `config.py`'s codex-id coercion (open question 2).

## Surfaces

Deleted:

| Path | Lines | Note |
|---|---|---|
| `scufris/api/legacy_agent.py` | 515 | the `/api/agent/*` router |
| `scufris/db/legacy/` (`__init__`, `gate`, `loaders`) | 740 | the one-way JSON import |
| `tests/test_legacy_agent_router.py` | 533 | |
| `tests/test_db_legacy.py` | 619 | |
| `examples/state_migration.py` | 232 | |
| `scufris/db/migrations/versions/*.py` | 5 files | replaced by one baseline |
| `LegacyImportRow` in `scufris/db/models.py:336-354` | 19 | table `legacy_import` |

Edited - the ones the Steps name:

| File | Why |
|---|---|
| `scufris/app.py:37,511` | drop the import and the `include_router` block |
| `scufris/db/__init__.py:20,26,30,117` | drop the re-exports and the `import_legacy_state` call in `open_state_database` |
| `CHANGELOG.md` | a BREAKING entry under `[Unreleased]` |

Edited - the ones the Steps do NOT name and a `rg` sweep finds:

| File | Why |
|---|---|
| `tests/conftest.py:196` | `"scufris.api.legacy_agent"` is a listed `get_backend` bind site; `monkeypatch.setattr` raises on a missing attribute, so this fails loudly across the whole suite |
| `tests/test_settings_store.py:192` | `test_writable_keys_match_the_api_update_model` imports `AgentConfigUpdate` from the router - see open question 3 |
| `tests/test_app.py:2092` | `test_legacy_agent_routes_delegate_to_scoped_diagnostics` |
| `tests/test_projects.py:30,326` | imports `LegacyImportRefused` from `scufris.db` and asserts a refusal |
| `tests/test_db_state_boundary.py:39,466` | imports `LegacyImportRow`, reads the gate rows |
| `tests/test_db_migrations.py:488-495` | `test_declared_tables_are_the_only_ones` lists `legacy_import` |
| `examples/host_agent.py:190-192` | monkeypatches `scufris.api.legacy_agent.get_backend` |
| `scufris/api/agent_runs.py:11` | docstring reference |
| `scufris/settings_store.py:22` | docstring reference |
| `README.md:181` | points at `examples/state_migration.py` |
| `scufris/README.md:315,362,457,550,607,665-680,742` | the module map, the delegation claim, the error table, the whole "Reading the legacy JSON in" section |

## Data and interfaces

Removed from `scufris.db`'s public surface:

```python
LegacyImportRefused    # RuntimeError subclass
import_legacy_state(db: Database, state_dir: Path) -> None
```

`open_state_database` loses its third step:

```python
def open_state_database(state_dir: Path) -> Database:
    db = open_database(state_dir)
    try:
        upgrade_to_head(db)
-       import_legacy_state(db, state_dir)
    except BaseException:
        db.close()
        raise
    return db
```

The migration chain today - all five deleted, replaced by one:

```
8f8087f3cc9c create_projects            (down_revision = None)
  -> 9b6587dab793 create_legacy_import
    -> 380a27d7fddb create_agent_session_outcome_settings_
      -> 3a5161b39846 create_auth_session_schedule_digest_and_
        -> e054a39a5fae create_config_change      (head)
```

Surviving tables after the squash: `projects`, `agents`, `agent_session`,
`agent_session_history`, `agent_outcome`, `settings_override`, `reasoning_turn`,
`auth_session`, `schedule`, `digest`, `host_action`, `config_change`. Twelve;
`legacy_import` is the thirteenth and it goes.

## Sketches

Illustrative only.

```diff
# scufris/db/__init__.py
-from .legacy import LegacyImportRefused, import_legacy_state
 from .migrate import upgrade_to_head
```

```diff
# scufris/app.py
-from .api.legacy_agent import LegacyAgentDeps, build_legacy_agent_router
 ...
-    app.include_router(
-        build_legacy_agent_router(LegacyAgentDeps(...))
-    )
```

```diff
# CHANGELOG.md, under [Unreleased] / ### Removed
+- **BREAKING: `/api/agent/*` is removed.** Use `/api/agents/orchestrator/*`.
+- **BREAKING: the pre-database JSON import is removed, and existing databases
+  are not carried forward.** v0.2.0 starts from an empty state directory;
+  delete `scufris.db` before upgrading.
```

## Shape

Startup, before and after:

```
  before:  open_database -> upgrade_to_head -> import_legacy_state
                                                     |
                                              db/legacy/gate.py
                                              one transaction per source,
                                              writes a legacy_import row

  after:   open_database -> upgrade_to_head
```

The HTTP surface, before and after:

```
  before:  /api/agent/*   ----\
                               >---- the SAME AgentStore / AgentRunService /
           /api/agents/orchestrator/*      AgentDiagnostics / supervisor

  after:   /api/agents/orchestrator/*   (the only door)
```

## Consequences and open questions

Cost: an existing database becomes unopenable (open question 1), and the
migration history stops being a record of how the schema got here. Bought: the
packages built next start from a tree with no compatibility layer, and one
revision to reason about instead of five.

Forecloses: any upgrade path from v0.1.x data. That is the accepted v0.2.0
position (`20260801-154211`: "the old database is dropped, not migrated"), and
this task is where it becomes irreversible.

**Open questions for the planner.** 1 and 3 change what gets written.

1. **A v0.1.x database will FAIL to open, and nothing decides how.** Its
   `alembic_version` holds `e054a39a5fae`, which the new baseline does not know,
   so `upgrade_to_head` raises "Can't locate revision". Worse,
   `test_a_database_from_a_newer_scufris_is_refused_without_a_backup`
   (`tests/test_db_migrations.py:374`) shows there is already a refusal path for
   an unknown revision - the operator will hit it with a message about a NEWER
   scufris, which is the opposite of what happened. Three options:
   - Let it fail with the existing message. Cheapest, most confusing.
   - Detect an unknown revision and refuse with a v0.2.0-specific message telling
     the operator to delete the database. **Recommended** - it is ~10 lines in
     `migrate.py` and it is the only thing standing between the maintainer and a
     mystery on their own box.
   - Nothing, and rely on the CHANGELOG. Rejected: the CHANGELOG is not in front
     of the operator when the unit fails to start.

   Whichever, the DoD's "builds the schema from empty" does not cover it. Add a
   proof.

2. **There is no "legacy branch" in `scufris/config.py`.** The Steps say to
   remove one. `rg -i legacy scufris/config.py` finds only the codex MODE id
   coercion (`_coerce_legacy_backend`, `canonical_backend`, lines 107, 373-401),
   which folds `app_server`/`exec` to `codex` and has nothing to do with either
   surface being deleted here. It is still load-bearing for persisted agent rows.
   **Do not delete it**; correct the Step.

3. **`test_writable_keys_match_the_api_update_model` loses its subject.**
   `tests/test_settings_store.py:192` asserts `AgentConfigUpdate.model_fields ==
   WRITABLE_KEYS` - two hand-kept copies of one whitelist, and the router is
   where the API copy lives. Deleting the router deletes the guard. Check whether
   `api/agents.py` declares an equivalent update model; if it does, re-point the
   test at it. If it does not, the whitelist loses its cross-check and that
   should be a recorded consequence rather than a silent one.

4. **`tests/test_projects.py` asserts a legacy refusal.** Line 326 expects
   `LegacyImportRefused` from a project-store path. Deleting the exception class
   deletes the assertion; confirm the underlying behavior (a damaged
   `projects.json` next to a database) has no remaining meaning rather than
   assuming it.

5. **The squash must run after 20260803-214749.** The baseline is autogenerated
   against the surviving models, and `host_action` / `config_change` move into
   `scufris_hostctl` in that task. If this runs first the baseline is regenerated
   twice. The epic's priorities already order it correctly (p102 then p101) -
   just do not reorder them.

6. **`test_declared_tables_are_the_only_ones` and
   `test_schema_has_no_pending_autogenerate_diff` are the real gate.** Both live
   in `tests/test_db_migrations.py` and both read `Base.metadata`, which after
   the carve is only complete if `migrations/env.py` imports every package's
   models. The DoD names the second one; name the first too - it is what catches
   a stray table surviving the squash.

7. **`examples/host_agent.py` needs a new patch target.** It reaches into
   `scufris.api.legacy_agent.get_backend`. `tests/conftest.py` lists
   `scufris.api.agent_runs` and `scufris.orchestrator.runs` as the other bind
   sites; the example should use whichever one its flow actually goes through, and
   the fix should be verified by running the example, not by grep.

8. **`docs/scratch/` and the release gate.** `nix flake check` runs `tatr check`
   and `check_file_size.py`; neither is affected. `scripts/check-release-ready.sh`
   reads `CHANGELOG.md` - the BREAKING entries above are what it will want.
