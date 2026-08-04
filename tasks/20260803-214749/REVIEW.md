# Review: Move the host control client into packages/hostctl

- TASK: 20260803-214749
- BRANCH: refactor/hostctl-package

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

The carve delivers the Story and the two guard tests are real: both sides of
`test_every_package_model_is_registered` are derived independently, and
`test_no_package_imports_a_sibling_private_module` exercises its new exemption
rather than carrying it as dead code. All seven Definition of Done proofs were
re-run in-session and pass on their stated criterion. The test split lost no
coverage: `test_nixos_config_change.py`'s 17 test names are exactly preserved
across the 8/9 split, `test_eventbus.py` is 5 for 5, and the supervisor split
keeps all 12 of master's names and adds 3 agent-level ones. The close-out's
numbers reproduce (1124 passed / 1 skipped, mypy clean on 244 files, 12 tests
in the package suite, all four nix builds exit 0).

What blocks the round is the exemption's justification and one false claim in
the new package README.

- [x] R1.1 (MAJOR) tests/test_package_boundaries.py:202 - the `SCHEMA_ASSEMBLY`
  exemption rests on a false premise and is avoidable. The comment claims
  `env.py` "CANNOT go through a facade" because the row classes are private,
  but `env.py` needs the IMPORT SIDE EFFECT, not the classes. Verified in the
  worktree: `import scufris_hostctl` alone puts both `host_action` and
  `config_change` into `Base.metadata`, because the facade reaches `.models`
  through `actions.py:32` and `hostconfig/changes.py:25`. Carving a permanent
  hole into the boundary rule this task exists to install, on a premise that
  does not hold, is the cost. Change `scufris/db/migrations/env.py:27` to
  `import scufris_hostctl  # noqa: F401 - registers this package's tables`,
  then delete `tests/test_package_boundaries.py:202-209` (the comment and the
  frozenset) and the skip branch at `:230-231`. The concept "boundary rule with
  an exemption list" disappears entirely;
  `test_every_package_model_is_registered` reads `env.py`'s imports with `ast`
  and keeps working unchanged, so the guard against a dropped import survives.
  - Response: fixed. `scufris/db/migrations/env.py:27` now imports `scufris_hostctl`;
    `SCHEMA_ASSEMBLY` and the skip branch are deleted, so the rule has no
    exemptions. The premise was indeed false - verified by importing the facade
    alone and printing `Base.metadata.tables` (both `host_action` and
    `config_change` present). `test_every_package_model_is_registered` still
    passes unchanged. Recorded as DECISION.md point 6.

- [x] R1.2 (MAJOR) packages/hostctl/src/scufris_hostctl/README.md:107 - "with
  no root, no real socket and a temporary database" is false, and the same
  commit contradicts it twice: `examples/hostctl_approval_flow.py:23` says "The
  helper runs in-process on a temporary unix socket - a real one, because
  `HostdClient` has no other transport and an example that faked it would prove
  nothing", and `DECISION.md` point 5 records exactly that reversal. The
  package's own public README is the one surface a cold reader trusts for what
  the example proves, and it carries the abandoned plan wording. Replace "no
  real socket" with the true claim, e.g. "against a fake executor over a
  temporary unix socket, with no root, no network and no NixOS machine".
  - Response: fixed. README now reads "against a fake executor over a temporary unix
    socket, with no root, no network, no NixOS machine and a temporary
    database". The neighbouring paragraph's `scufris_hostctl.models` claim was
    stale after R1.1 and is corrected in the same edit.

- [x] R1.3 (MINOR) packages/hostctl/src/scufris_hostctl/actions.py:193 - the
  moved module still cross-references its pre-move path:
  `:class:`~scufris.host_approvals.HostApprovalService``. It is the only
  `:class:`~scufris.*`` reference left anywhere under `packages/`. Change it to
  `:class:`~scufris_hostctl.HostApprovalService``.
  - Response: fixed. Now `:class:`~scufris_hostctl.HostApprovalService``.

- [x] R1.4 (MINOR) packages/core/src/scufris_core/eventbus.py:23 - stale
  pre-move module names in four live files the rename left behind. Here
  ``scufris.hostclient`` -> `scufris_hostctl.client`. Same fix at
  `packages/hostd/src/scufris_hostd/__init__.py:5` ("through
  ``scufris.hostclient``"),
  `packages/hostd/src/scufris_hostd/actions/__init__.py:31` ("resolved
  (``scufris/hostconfig``)"), and `examples/hostd_socket_roundtrip.py:26`
  ("rather than `scufris.hostclient`").
  - Response: fixed, all four: `scufris_hostctl.client` in `eventbus.py` and
    `scufris_hostd/__init__.py`, `scufris_hostctl.hostconfig` in
    `scufris_hostd/actions/__init__.py`, and `scufris_hostctl.client` in
    `examples/hostd_socket_roundtrip.py`.

- [x] R1.5 (MINOR) pyproject.toml:31 - "scufris/db/models.py still declares
  thirteen tables" is now wrong. Two row classes left for `hostctl`, so
  `grep -c '^class .*(Base):' scufris/db/models.py` is 11, down from 13 on
  master. The same stale count appears at `tests/test_package_boundaries.py:17`
  ("beside thirteen row classes") and `:92` ("like thirteen ordinary classes").
  Change all three to eleven.
  - Response: fixed. All three now say eleven; `grep -c '^class .*(Base):'
    scufris/db/models.py` is 11.

- [x] R1.6 (NIT) packages/hostctl/tests/test_config_change_service.py:38 -
  `models,  # noqa: F401 - registers the tables on Base` is a no-op whose
  comment states something untrue: the very same
  `from scufris_hostctl import (...)` statement imports the facade, which has
  already registered both tables (see R1.1). Drop the `models` entry and its
  comment. Same at `examples/hostctl_approval_flow.py:52`.
  - Response: fixed. Both `models` entries and their comments dropped.

- [x] R1.7 (NIT) packages/hostctl/src/scufris_hostctl/models.py:23 - the moved
  docstrings carry task IDs (`20260801-100405 DECISION.md 3` and
  `20260801-100413 DECISION.md 4` here, `20260803-002141 DECISION.md 1` at
  `:62`), which `AGENTS.md:107` forbids: "Task IDs belong in task records and
  Markdown, never in code comments or docstrings". Inherited from the verbatim
  move the plan asked for, but the file is new, so the rule applies to it now.
  Strip the IDs and keep each invariant as a bare fact.
  - Response: fixed for `models.py`: three IDs stripped, each invariant kept as a
    bare fact. Not widened - the same pattern is repo-wide (`changes.py:75`,
    `service.py:141`, `core/engine.py:120`, `core/supervisor.py:101`, and a
    dozen root modules), all of it pre-existing prose this task only moved.
    Cleaning it up is its own task, not this branch's.

- Process signal: the plan's "no real socket" clause for the example was not
  achievable, and the implementation correctly reversed it in `DECISION.md`
  point 5 - but the reversal did not propagate to the package README written in
  the same commit (R1.2). A plan clause that gets overturned mid-task needs a
  sweep of the prose written against it, not just a decision record.

### Verification

- Re-run in-session: `uv run python -m pytest` (1124 passed, 1 skipped),
  `uv run mypy .` (no issues, 244 files), `ruff check` and
  `ruff format --check` (clean).
- All seven DoD proofs re-run and passing on their stated criterion, including
  `nix flake check`, `nix build .#scufris .#scufris-web .#scufris-hostd` and
  `nix build .#scufris-vm-test`, all exit 0.
- Test-split coverage re-derived by diffing test-function names against
  `master`: `test_nixos_config_change.py` 17 -> 8 + 9, identical name set;
  `test_eventbus.py` 5 -> 5; `test_supervisor.py` 12 -> 3 + 12, a superset that
  adds `test_a_completed_agent_turn_settles_done`,
  `test_a_failed_run_publishes_a_streamerror` and
  `test_a_terminal_error_event_is_recorded_on_run_error`. Nothing weakened or
  deleted.
- R1.1 re-derived independently of the reviewer, by importing the facade alone
  and printing `Base.metadata.tables`.
- No `manual:` proofs in the Definition of Done, so there are no pending user
  checks.

## Round 2

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

All seven round-1 findings are confirmed fixed and ticked. R1.1's design change
holds: `env.py:27` is `import scufris_hostctl`, `SCHEMA_ASSEMBLY` and its skip
branch are gone, and the boundary rule now ships with no exemption arm. R1.7's
scoping pushback is accepted - the task-ID pattern is pre-existing in modules
this branch only moved. No regression from any of the seven fixes; the full
suite, mypy, ruff and all seven Definition of Done proofs re-run green,
including the three nix builds.

What blocks the round is the guard round 1 assumed would survive R1.1, checked
this time by breaking it.

- [x] R2.1 (MAJOR) tests/test_db_migrations.py:601 -
  `test_every_package_model_is_registered` is vacuous in the run that gates the
  branch. It populates `Base.metadata` by importing `_env_imports()`'s names
  into the CURRENT interpreter, but by the time it runs under a full `pytest`,
  `scufris_hostctl` is already in `sys.modules` - the app, the example test and
  the package suite all import it - so the tables are registered whatever
  `env.py` says. Re-derived in-session, independently of the reviewer: with
  `scufris/db/migrations/env.py:27` deleted, `uv run python -m pytest` is
  1124 passed / 1 skipped, exit 0, while
  `pytest tests/test_db_migrations.py::test_every_package_model_is_registered`
  alone goes red. The test is new on this branch (`b014db2`), so this is the
  diff's own problem, not a pre-existing one, and the failure it exists to
  catch - a package whose tables are never created - reaches master green.
  Build the metadata in a fresh interpreter instead: `subprocess.run([
  sys.executable, "-c", script], check=True, capture_output=True)` where
  `script` imports only the `_env_imports()` names and prints
  `sorted(Base.metadata.tables)`, then assert against that. The test then
  measures `env.py`'s imports rather than whatever the session already did.
  - Response: fixed, and the fix is falsified rather than argued.
    `tests/test_db_migrations.py` now builds `Base.metadata` in a `python
    -c` child given exactly `_env_imports()`'s names
    (`_tables_env_registers`), so the check sees only what `env.py` imports.
    Re-derived the same way the finding was: with
    `scufris/db/migrations/env.py:27` deleted, `uv run python -m pytest` is
    now 1 failed / 1123 passed - the guard bites in the canonical gate - and
    green again at 1124 passed / 1 skipped once restored. Recorded as
    DECISION.md point 7.

- Process signal: round 1 accepted "the guard against a dropped import
  survives" as reasoning rather than by mutating `env.py` and re-running the
  canonical gate. A claim that a test still guards something is only worth what
  breaking the guarded thing proves.
- Process signal: TASK.md's close-out numbers and round 1's verification
  section both reproduce exactly. No fabricated evidence found.

### Verification

- Full suite 1124 passed / 1 skipped; `mypy` clean on 244 files; `ruff check`
  and `ruff format --check` clean.
- All seven proofs re-run on their stated criterion, including `nix flake
  check`, `nix build .#scufris .#scufris-web .#scufris-hostd` and
  `nix build .#scufris-vm-test`, all exit 0.
- R2.1 re-derived in-session by deleting `env.py:27`, running the full suite
  (green) and the registration test alone (red), then restoring the file.
- `test_nixos_config_change.py`'s test-name set re-checked as identical to
  master's across the split.
- No `manual:` proofs, so there are no pending user checks.

## Round 3

- REVIEWER: out-of-context
- VERDICT: APPROVE

R2.1 is confirmed fixed and ticked. `_tables_env_registers` builds the metadata
in a `python -c` child given exactly `_env_imports()`'s names, and the guard
bites where it has to: re-derived independently by deleting
`scufris/db/migrations/env.py:27`, running the full suite (1 failed / 1123
passed, failing on `{'scufris_hostctl': ['config_change', 'host_action']}`) and
restoring it (1124 passed / 1 skipped, clean tree). The helper survives a
foreign cwd, cannot silently return an empty set, and the round-2 close-out's
numbers reproduce exactly.

One MINOR remains open. It does not block: the guard is correct, and the finding
is about what it prints when a DIFFERENT defect makes the child fail.

- [ ] R3.1 (MINOR) tests/test_db_migrations.py:609 - `check=True` together with
  `capture_output=True` swallows the child's traceback. Re-derived in-session:
  the raised `CalledProcessError` renders as `Command '[...]' returned non-zero
  exit status 1.` and nothing else, so if a package's `models` module ever
  raises on import this guard goes red with no clue why and the maintainer has
  to rebuild the `python -c` by hand. Use `check=False` and assert the child's
  own output instead: `assert completed.returncode == 0, completed.stderr`
  before parsing stdout.
  - Response:

- Process signal: round 2's fix was falsified by mutation before it was
  claimed, and the round-2 close-out numbers reproduce exactly. The practice
  round 1 skipped - break the guarded thing, do not argue that the guard holds
  - is what produced a real finding in round 2 and a clean confirmation here.

### Verification

- Full suite 1124 passed / 1 skipped, exit 0; `mypy` clean on 244 files;
  `ruff check` and `ruff format --check` clean.
- All seven proofs re-run on their stated criterion, including `nix flake
  check`, `nix build .#scufris .#scufris-web .#scufris-hostd` and
  `nix build .#scufris-vm-test`, all exit 0.
- R3.1 re-derived in-session by running the same call shape against a failing
  child and reading the exception's message.
- Doc-surface sweep for pre-move paths (`scufris.hostclient`,
  `scufris.host_actions`, `scufris.host_approvals`, `scufris.hostconfig`): only
  `tasks/` (exempt) and CHANGELOG.md's deliberate rename note.
- No `manual:` proofs, so there are no pending user checks.
