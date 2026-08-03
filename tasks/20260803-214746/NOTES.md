# Notes: Bootstrap the uv workspace and the core package

Goal in one line: turn the repo into a two-member `uv` workspace whose first
member, `scufris-core`, holds the database machinery and nothing domain-specific
- and prove the boundary with tests rather than a README.

## What changes

Nothing an operator sees. `scufris` starts, migrates and serves exactly as it
does today; the wheel still ships one console script pair and the same
migrations.

What a MAINTAINER sees:

| Before | After |
|---|---|
| one distribution `scufris`, one import root `scufris/` | workspace: `packages/core` (`scufris_core`) + root `scufris` |
| "core is generic" is a claim | `test_core_is_domain_free` |
| "packages do not reach into each other" is a claim | `test_no_package_imports_a_sibling_private_module` |
| no gated runnable proof per package | `examples/core_unit_of_work.py` + `tests/test_examples.py` |
| `flake.nix:74` `members` commented out | set, `uv.lock` regenerated |

## Surfaces

Moves into `packages/core/src/scufris_core/`:

| File | Why |
|---|---|
| `scufris/db/engine.py` (327 ln) | the whole transactional boundary: `Database`, `transaction()`, pragmas, `BEGIN IMMEDIATE`, nesting guard, loop-thread refusal. Domain-free already |
| `Base` out of `scufris/db/models.py` | the shared `DeclarativeBase` every package will register rows against. Only the class, 2 lines plus docstring; the 12 row classes stay |

New:

| File | Why |
|---|---|
| `packages/core/pyproject.toml` | member metadata + its own narrow dependency list |
| `packages/core/src/scufris_core/__init__.py` | the public facade; the ONLY surface a sibling may import |
| `examples/core_unit_of_work.py` | the primary proof: temp SQLite, two rows, one rollback, offline |
| `tests/test_examples.py` | the harness every later package plugs into |
| `tests/test_boundaries.py` (name TBD) | the two enforcement tests |
| `tasks/20260803-213242/DECISION.md` | the five decisions, recorded before four packages depend on them |

Edited:

| File | Why |
|---|---|
| root `pyproject.toml` | `[tool.uv.workspace] members`, `[tool.uv.sources] scufris-core = {workspace = true}`, dependency on `scufris-core` |
| `flake.nix:74` | `members = ["scufris"]` on the editable overlay; verify `workspace.deps.default` still resolves the app venv |
| `uv.lock` | regenerated |
| `scripts/check_file_size.py` | `COVERED_ROOTS` gains `packages` - see open question 6 |
| `scufris/db/__init__.py`, `migrate.py`, `models.py`, `migrations/env.py` | re-point at `scufris_core` |
| ~4 importers of `Database` / `db.models` | mechanical; `rg 'from .*db import|db\.engine'` |
| `scufris/README.md` s9, `AGENTS.md` sources table | the module map moves |

Not touched: `scufris/db/models.py` rows, `migrations/versions/*`, `alembic.ini`,
`scufris/db/legacy/` (deleted by 20260803-214750), `web/`, every domain package.

## Data and interfaces

`scufris_core` public API - the whole of it:

```python
DATABASE_FILENAME: str
FILE_MODE: int
class Base(DeclarativeBase): ...
class Database:
    engine: Engine
    path: Path
    def transaction(self) -> ContextManager[Connection]: ...
    def close(self) -> None: ...
def database_path(state_dir: Path) -> Path: ...
def open_database(state_dir: Path) -> Database: ...
```

Nothing else. `open_state_database` / `state_database` / `close_*` stay in
`scufris/db/__init__.py`: they compose open + migrate + legacy-import, and two of
those three are root-owned (see open question 4).

The two enforcement tests:

```python
def test_core_is_domain_free() -> None: ...
    # scufris_core imports no sibling; no module under it declares a
    # __tablename__; the only Base subclass it defines is Base itself.
def test_no_package_imports_a_sibling_private_module() -> None: ...
    # AST-walk every package's imports; a package may name a sibling's
    # distribution root, never `<sibling>.models` or `<sibling>.repo`.
```

## Sketches

Illustrative only.

```diff
# scufris/db/models.py
-from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
-
-class Base(DeclarativeBase):
-    """The metadata Alembic compares a database against."""
+from sqlalchemy.orm import Mapped, mapped_column
+from scufris_core import Base
```

```diff
# pyproject.toml (root)
+[tool.uv.workspace]
+members = ["packages/*"]
+
+[tool.uv.sources]
+scufris-core = { workspace = true }
 dependencies = [
-    "sqlalchemy>=2.0",
+    "scufris-core",
```

```diff
# flake.nix:74
-      # members = [ "scufris" ];
+      members = ["scufris"];
```

## Shape

```
                 packages/core  ->  scufris_core
                 (engine, Database.transaction, Base)
                        ^                    ^
                        |                    |
        scufris/db/models.py         scufris/db/migrate.py
        (12 row classes, ->Base)     (alembic, script_location=
                        ^             scufris.db.migrations)
                        |                    |
                  scufris/db/__init__.py: open_state_database
                       open -> upgrade -> import_legacy
                                 |
                          stores, routers, CLI

  dependency arrows point INTO core only. core imports no sibling.
```

The wheel/venv shape after the change:

```
  uv workspace root
   |- pyproject.toml   -> scufris        (depends on scufris-core)
   `- packages/core    -> scufris-core   (sqlalchemy only)
  nix: workspace.deps.default resolves both; editable overlay
       members=["scufris"] keeps the app editable in the dev shell
```

## Consequences and open questions

Cost: one more `pyproject.toml`, one more lock entry, one more hop for a reader
following `Database` to its definition, and a `uv.lock` churn that every open
branch will conflict on. Bought: the boundary becomes falsifiable, and the four
later carves are mechanical repeats of a path already proven.

Forecloses nothing structural - a member can be dissolved back into the root by
deleting its `pyproject.toml` and moving the tree.

**Open questions for the planner.** 1, 2 and 5 change what gets written.

1. **`scufris/enums.py` is NOT domain-free, and the task says to move it.**
   Every symbol in its 138 lines belongs to a future sibling: `ORCHESTRATOR_ID`,
   `HOST_AGENT_ID`, `Audience`, `audience_for`, `Backend`, `PermissionMode`,
   `AgentState`, `RunPhase` -> `agents`; `AuthMode` -> `agents`/auth;
   `AuthPolicy` -> the composition root. There is no generic enum in the file.
   Moving it wholesale is the exact junk-drawer decay the epic's Sequencing
   section names, and it should fail `test_core_is_domain_free` if that test is
   written honestly. **Recommendation: leave `enums.py` at the root** and let
   each enum travel with its package when that package is built (20260729-102157
   for `agents`). Assumption taken here unless the planner overrides it.

2. **Three of the seven things `core` is said to own do not exist.**
   - `ids`: `python-ulid` is declared in `pyproject.toml` and imported by
     ZERO files (`rg ulid scufris/ tests/ examples/` -> nothing). Creating
     `scufris_core.ids` is speculative; per YAGNI, do not.
   - `time`: no such module, no candidate helper.
   - error types: there is no `scufris/errors.py`. Errors are local and
     domain-specific (`orchestrator/errors.py`, `hostclient.py`,
     `opencode_client.py`). Nothing generic to hoist.
   - "the session factory": the codebase uses SQLAlchemy **Core** - there is no
     `sessionmaker` and no ORM `Session` anywhere. The unit of work IS
     `Database.transaction()` yielding a `Connection`. The epic's wording should
     be read as that, not as a missing component to build.

   So `core` = engine + `Database` + `Base`. That is smaller than the epic's
   table implies, and smaller is the point.

3. **`Base` moves but its rows do not**, which means `scufris/db/models.py`
   imports `scufris_core` - correct direction - and `migrations/env.py` keeps
   importing `scufris.db.models` for the metadata. Fine with one member; the
   `test_every_package_model_is_registered` guard is the epic's, not this task's.

4. **Where does `migrate.py` land?** It is Alembic-coupled and its
   `script_location` is `importlib.resources`-resolved inside
   `scufris.db.migrations`. Recommendation: it stays at the root, so `core` keeps
   no alembic dependency at all and `core`'s declared deps drop to
   `sqlalchemy` alone (the task's list of five is too wide -
   `alembic`/`pydantic`/`pydantic-settings`/`python-ulid` are all unused by what
   actually moves). Same for `scufris/db/legacy/`, which 20260803-214750 deletes.

5. **`test_no_package_imports_a_sibling_private_module` is vacuous with one
   member.** With only `core` and the root there is no sibling pair to violate
   it, so "must fail for the right reason" cannot mean a real violation. It has
   to be written against the DECLARED graph - a fixture listing the eight future
   packages, asserted over whichever exist - or it will pass green for four
   tasks and only start working at 20260803-214747. Planner should pick which.

6. **`scripts/check_file_size.py` `COVERED_ROOTS = ("scufris", "tests",
   "web/src")`.** Moving a file into `packages/` silently exempts it from the
   600-line cap. `engine.py` is 327 lines so nothing is hiding today, but the
   gap must close in this task or the carve quietly disables the guard for four
   more packages.

7. **`tests/test_examples.py` needs a policy on the existing eleven examples.**
   Several boot a real uvicorn (`auth_session.py`) or need a real NixOS box
   (`host_inspect.py`, `nixos_change.py`). 20260801-154211 says manual ones are
   "marked so the gate skips them" - that marker does not exist yet, and this
   task builds the harness. Cheapest honest version: the gate runs an explicit
   opt-in list, starting with `core_unit_of_work.py`.

8. **DECISION.md path.** The Steps say "Write `DECISION.md`"; the Definition of
   Done checks `tasks/20260803-213242/DECISION.md` - the EPIC folder. Following
   the DoD. `tatr context` reports `tasks/20260803-214746/DECISION.md` missing,
   which is expected and not a failure.

9. **Root wheel packaging.** `[tool.hatch.build.targets.wheel] only-include =
   ["scufris"]` stays correct for the root; `packages/core` needs its own
   build-backend block, and `nix build .#scufris` must be re-run to prove the
   runtime venv still resolves both members (DoD: "the build is unchanged").
