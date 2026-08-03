# Bootstrap the uv workspace and the core package

- PRIORITY: 105
- TAGS: refactor, v0.2.0, architecture, packaging
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -
- PARENT: 20260803-213242

## Story

As the Scufris maintainer, I want the `uv` workspace to exist with one real
member - `core` - so that the packaging machinery is proven against the smallest
possible package before four more move into it.

`core` holds the database machinery and nothing domain-specific: the engine, the
session factory, `Base`, the unit of work, ids, time, enums, settings and error
types. It imports no sibling and declares no domain table.

## Steps

- [ ] Write `test_core_is_domain_free` and `test_no_package_imports_a_sibling_private_module`
      FIRST, against the empty workspace. Both must fail for the right reason.
- [ ] Add `[tool.uv.workspace] members = ["packages/*"]` to the root
      `pyproject.toml`.
- [ ] Create `packages/core/` with its own `pyproject.toml` declaring only
      `sqlalchemy`, `alembic`, `pydantic`, `pydantic-settings`, `python-ulid`.
- [ ] Move `scufris/db/engine.py`, `Base`, the session factory, `scufris/enums.py`
      and the error types into `scufris_core`. Leave `scufris/db/models.py` where
      it is: its rows are domain tables and belong to the packages that own them.
- [ ] Re-point the root `pyproject.toml` to depend on `scufris-core`, and update
      every import in `scufris/`.
- [ ] Set `members` in the flake's `mkEditablePyprojectOverlay` (`flake.nix:74`)
      and regenerate `uv.lock`.
- [ ] Add `examples/core_unit_of_work.py`: open a temporary SQLite database,
      write two rows in one transaction, roll one back, print the result. No
      host, no provider, no network.
- [ ] Add `tests/test_examples.py`, which runs every offline example and fails on
      a non-zero exit. This is the harness the later packages plug into.
- [ ] Write `DECISION.md` recording the ten-unit cut, the public-API-only import
      rule, the rejection of Protocol ports, distributed tables over central
      CRUD, and one Alembic history.

## Definition of Done

- `core` imports on its own and depends on no sibling
  (cmd: `uv run python -c "import scufris_core"`).
- `core` declares no domain table and imports no sibling
  (test: `test_core_is_domain_free`).
- The boundary rule is enforced rather than documented
  (test: `test_no_package_imports_a_sibling_private_module`).
- The example runs green offline and is gated
  (cmd: `python -m pytest tests/test_examples.py -k core`).
- The build is unchanged in behavior
  (cmd: `nix flake check && nix build .#scufris`).
- The decisions behind the carve are recorded before four packages depend on
  them (cmd: `test -f tasks/20260803-213242/DECISION.md`).

## Notes

- Parent: 20260803-213242. Read its Epic section before starting; it carries the
  dependency graph, the boundary rule and the Alembic mechanics.
- `uv2nix.lib.workspace.loadWorkspace` is already the loader (`flake.nix:61`).
  This task turns on a knob that is already there, commented, at `flake.nix:74`.
- Keep `core` small enough that its contents are obvious. If something is
  arguably domain-specific, it does not go here.
