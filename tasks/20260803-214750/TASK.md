# Delete the legacy agent router and JSON import, squash to one baseline revision

- PRIORITY: 101
- TAGS: refactor,v0.2.0,architecture,storage
- KIND: TASK
- ACTIVITY: -
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
- [ ] Remove the legacy router from `scufris/app.py` and the legacy branch from
      `scufris/config.py`.
- [ ] Delete all five revisions under `scufris/db/migrations/versions/` and
      generate ONE baseline revision by autogenerate against the surviving
      models.
- [ ] Confirm `alembic upgrade head` builds the schema from empty, and that the
      pending-autogenerate-diff test is green against the new baseline.
- [ ] Note in `CHANGELOG.md` that the legacy agent API and the JSON import path
      are removed and that existing databases are not carried forward.

## Definition of Done

- No legacy surface survives
  (cmd: `! rg -q 'legacy_agent|db\.legacy|legacy_import' --glob '!tasks/**' --glob '!CHANGELOG.md' .`).
- Exactly one migration revision exists and it builds the schema from empty
  (cmd: `test $(ls scufris/db/migrations/versions/*.py | wc -l) -eq 1`).
- The schema and the revision do not disagree
  (test: `test_schema_has_no_pending_autogenerate_diff`).
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
