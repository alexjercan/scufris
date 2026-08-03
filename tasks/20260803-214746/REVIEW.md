# Review: Bootstrap the uv workspace and the core package

- TASK: 20260803-214746
- BRANCH: refactor/uv-workspace-core

## Round 1

- REVIEWER: out-of-context (two lanes: behavior/proofs + correctness; design/standards/docs)
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (BLOCKER) .github/workflows/release.yaml:249 - the root wheel now
  carries `Requires-Dist: scufris-core`, a distribution that exists on no
  registry, while `uv build` still produces only the root wheel, so the release
  job's clean-venv smoke test cannot resolve it and the published artifact would
  be uninstallable. Reproduced on this branch: `uv build` emits
  `scufris-0.1.0-py3-none-any.whl` whose METADATA carries
  `Requires-Dist: scufris-core`, and `uv pip install` of that wheel into a fresh
  venv fails with "Because scufris-core was not found in the package registry
  ... requirements are unsatisfiable". Change line 235 to
  `uv build --all-packages`, replace the single-wheel `ls` at line 249 with an
  install of both wheels (`uv pip install --python /tmp/smoke/bin/python
  dist/scufris_core-*.whl dist/scufris-*.whl`), and pin the root dependency at
  `pyproject.toml:22` to `scufris-core==0.1.0` so the artifact names a
  resolvable version.

  Response: fixed in 038b332, except the pin - see below. `uv build` became
  `uv build --all-packages`, and the smoke step installs the whole wheel set in
  ONE resolution (`uv pip install ... dist/*-py3-none-any.whl`) rather than the
  single-wheel `ls`. The glob is deliberate: `dist/*.whl` is already what the
  "Attach the distribution" step uploads, so the smoke test now exercises
  exactly the artifact set the release publishes, and a third member added later
  needs no edit here.

  Both directions re-derived on this branch: root-only `uv build` + install of
  that one wheel still fails ("scufris==0.1.0 depends on scufris-core ...
  requirements are unsatisfiable"); `--all-packages` + install of both reports
  `scufris 0.1.0`.

  PUSHBACK on `scufris-core==0.1.0`. The release attaches wheels to a GitHub
  release; it publishes to no registry, so `Requires-Dist: scufris-core` is
  resolvable exactly when the member wheel is installed alongside - which the
  fixed smoke step now proves. A literal `==0.1.0` would be a THIRD version
  string, unchecked by anything, and it breaks the moment the member is bumped -
  which is the same hazard R1.12 raises from the other side. Fixed at the root
  instead: `check_agreement` now takes `member_texts` and fails when any
  `packages/*/pyproject.toml` names a version other than the root's, so the
  members are a matched set by construction and the unpinned dependency always
  resolves within it. See R1.12.
- [x] R1.2 (MAJOR) scufris/db/__init__.py:11 - the move deleted
  `scufris/db/engine.py` but left eleven in-source pointers at that path, so
  every "the rules are in ..." reference is now a dead path a cold reader
  follows to nothing. Retarget all of them to
  `packages/core/src/scufris_core/engine.py`: `scufris/db/__init__.py:11`,
  `scufris/db/migrate.py:99`, `scufris/api/auth.py:109`,
  `scufris/api/agent_runs.py:145`, `scufris/auth/store.py:7`,
  `scufris/auth/store.py:54`, `scufris/orchestrator/turn.py:48`,
  `scufris/orchestrator/runs.py:155`, `scufris/agent_store/registry.py:12`,
  `scufris/scheduler.py:78`, `scufris/wake.py:94`. In the same edit rewrite
  `scufris/db/__init__.py:3`, whose first paragraph still says "``engine`` owns
  the boundary" as if `engine` were a submodule of `scufris.db` - directly
  contradicting the `scufris/README.md:497` line this diff wrote ("The boundary
  itself is `scufris_core`"); name `scufris_core` as the boundary there and list
  only `models`/`migrate`/`legacy` as this package's parts.

  Response: fixed in 038b332. All eleven retargeted; verified by the absence
  proof rather than by counting - `rg -n 'scufris/db/engine\.py'` outside
  `tasks/` now returns nothing, and the new path appears at twelve sites (the
  eleven, plus `examples/core_unit_of_work.py`, which already named it
  correctly). Nine paragraphs were re-wrapped: the longer path pushed them past
  the column the surrounding prose keeps. `scufris/db/__init__.py`'s opening is
  rewritten - the boundary is `scufris_core`, in another distribution, and this
  package is the application's half of persistence (`models` against
  `scufris_core.Base`, `migrate`, `legacy`).
- [x] R1.3 (MAJOR) tests/test_package_boundaries.py:159 - `_import_roots()`
  globs only `packages/*/src/*`, so the `scufris` distribution - a declared
  member in `uv.lock`'s `[manifest] members` and by far the largest consumer of
  `scufris_core` - is exempt from the sibling-private-import rule, while
  `tasks/20260803-213242/DECISION.md:54` states the test "AST-walks every
  member's imports". Add the root to the mapping
  (`{"scufris": REPO_ROOT / "scufris"}`, which passes today), or narrow both
  `DECISION.md:54` and `scufris/README.md:458` to "every `packages/` member".

  Response: fixed in 038b332, taking the first option - `_import_roots()` now
  adds `{"scufris": REPO_ROOT / "scufris"}`, so `DECISION.md:54` and
  `scufris/README.md:458` become true as written rather than narrowed to match
  a weaker check. It passes today: no module under `scufris/` names a
  `scufris_core` submodule (`rg 'scufris_core\.' --glob '*.py'` finds only
  prose, plus `tests/test_logsetup.py`, which R1.4 fixes). The root is also the
  tree the carve moves code OUT of, so it is the one most likely to keep an
  import pointing at where a module used to sit - the docstring now says so.
- [x] R1.4 (MINOR) tests/test_logsetup.py:7 - the re-point reaches around the
  facade, importing `configure_logging`, `new_request_id`, `set_request_id` and
  `truncate` from the private submodule `scufris_core.logsetup` - the exact
  pattern `packages/core/src/scufris_core/__init__.py:17` and
  `test_no_package_imports_a_sibling_private_module` declare wrong. Import the
  four public names from `scufris_core` and keep only `_RequestIdFilter` on the
  submodule line, with a one-line note that it is a deliberate private-name
  test.

  Response: fixed in 038b332. The four public names come from `scufris_core`;
  `_RequestIdFilter` stays on its own `from scufris_core.logsetup import` line
  under a two-line note saying it is private and reachable nowhere else.
- [x] R1.5 (MINOR) packages/core/src/scufris_core/__init__.py:13 - `AGENTS.md`
  line 106 ("Task IDs belong in task records and Markdown, never in code
  comments or docstrings") is violated by three new sites: this line's
  `tasks/20260803-214746/TASK.md`, `tests/test_package_boundaries.py:31`, and
  `tests/test_package_boundaries.py:193` ("20260803-214747's `hostd`"). Keep the
  invariant in prose ("logsetup is generic and shared by four future packages";
  "vacuous until a second member exists") and delete the IDs.

  Response: fixed in 038b332. All three IDs deleted, each replaced by the
  reason it was standing in for. The facade docstring now carries the logsetup
  justification inline (generic, 87 lines over `logging`/`uuid`/`contextvars`,
  shared across four packages, and no package can be a member while it imports
  a root module for logging) as the third bullet of the list, which also fixes
  the "Two things live here" count that had gone stale at three.

  Scoped to the three sites this diff introduced. `rg '\d{8}-\d{6}'` over
  `packages/`, `scufris/`, `tests/` and `examples/` still reports pre-existing
  IDs - eight in `examples/` docstrings, and `scufris_core/engine.py:120`,
  which arrived verbatim with the `git mv` (it is at `scufris/db/engine.py:120`
  on `master`). Not this branch's to fix, and not this branch's regression
  either.
- [x] R1.6 (MINOR) alembic.ini:26 - the comment names `scufris.logsetup`, a
  module this diff moved; change it to `scufris_core.logsetup`.

  Response: fixed in 038b332.
- [x] R1.7 (NIT) scufris/api/request_log.py:4 - the docstring's three bare
  `logsetup` mentions (lines 4, 5, 8) now name a module in another distribution,
  while `scufris/README.md:321` was qualified to `scufris_core.logsetup` in this
  same diff; qualify all three the same way.

  Response: fixed in 038b332.
- [x] R1.8 (NIT) tests/test_package_boundaries.py:124 - `_domain_free` takes an
  `allowed` parameter but its failure message hardcodes `CORE_MODULES`, so the
  falsifier arm would print advice about `core` when pointed at `scufris/db`;
  say "add it to the allowlist this check was called with" instead.

  Response: fixed in 038b332, with the wording as suggested.
- [x] R1.9 (NIT) tests/test_package_boundaries.py:117 - `classes` and `owner`
  are keyed by bare class name across the whole tree, so two same-named classes
  in different modules collide and one silently drops out of the declarative
  walk; key both dicts by `(module, name)` and match bases against the set of
  bare names.

  Response: fixed in 038b332. `classes` is now keyed by `(module, name)` and
  `_declarative_classes` returns those pairs, matching bases against the bare
  names reached so far (an `ast` base node carries no module, so bare is the
  only thing to match on). `owner` is gone - the key carries the module, so the
  second dict had nothing left to hold.
- [x] R1.10 (NIT) examples/core_unit_of_work.py:29 - the comment says "the root
  and the workspace member" but only the member's `src` goes on `sys.path`
  (correctly - the script imports no `scufris`); drop "the root and".

  Response: fixed in 038b332; the comment now states positively why only the
  member's `src` is needed.
- [x] R1.11 (NIT) tasks/20260803-214746/TASK.md:73 - the census says "twelve
  importers under `scufris/`" where `tasks/20260803-213242/DECISION.md:95` says
  eleven; the tree has eleven Python modules plus `scufris/README.md`. Correct
  the TASK.md line to eleven modules.

  Response: fixed in 038b332. Eleven, and the Step now PASTES the list rather
  than stating a number - `agent/appserver.py`, `agent_mcp_server.py`,
  `api/request_log.py`, `backends/claude.py`, `backends/codex.py`,
  `backends/opencode.py`, `cli.py`, `den_mcp_server.py`, `host_mcp_server.py`,
  `hostd/main.py`, `mcp_server.py`. Re-derived on the branch, not recounted by
  hand: seven module-level `from scufris_core import ...` lines naming a
  logsetup symbol, plus four function-local ones in the MCP server `main()`s.
  See the process-signal response below.
- [x] R1.12 (NIT) packages/core/pyproject.toml:3 - a second `version = "0.1.0"`
  now exists that `scripts/check-release-ready.sh` does not check, so a root
  bump silently leaves the member behind; note in `docs/RELEASING.md` that
  member versions are not released artifacts, or add them to the
  version-agreement check.

  Response: fixed in 038b332, taking the second option - a note alone would not
  have held, because R1.1's fix makes the member wheel a PUBLISHED artifact
  rather than a private detail. `release_tools.check_agreement` gained a
  `member_texts` argument and `member_pyprojects(root)` to supply it from
  `packages/*/pyproject.toml`; a member naming a version other than the root's
  now fails `python -m scripts.release_tools check`, and therefore
  `scripts/check-release-ready.sh` and the release guard job. Two tests:
  `test_every_workspace_member_shares_the_root_version` over the live tree and
  `test_a_member_left_behind_is_rejected` as its falsifier.
  `docs/RELEASING.md` gained the member to its "sources must agree" list, to
  step 1, and corrected "wheel and sdist build" to name
  `uv build --all-packages`.

Process signal:

- The Step census drifted a second time (twelve vs eleven `logsetup`
  importers), the same class as the "18 modules plus 25+ test files" error the
  plan already had to correct once. Counting importers by hand is the recurring
  failure; pasting the `rg -l` output into the record would end it.
- Two edits outside the Steps - `[tool.ruff.lint.isort] known-first-party` and
  `flake.nix`'s `export REPO_ROOT=$PWD` - are disclosed under Difficulties
  rather than quietly folded in. Correct handling, and the `REPO_ROOT` one
  exposed a real latent hole in the check sandbox.
- The DoD's build proof exercises the uv2nix path only. Nothing in the proof set
  touches `uv build`, which is exactly where R1.1 hides: the task that creates a
  second distribution carried no proof over the artifact the release job
  actually publishes.

Response to the process signal (round 1):

- Census drift: taken, and applied rather than acknowledged. The `logsetup`
  Step now carries the eleven names, not the number eleven. Two counts in this
  record have now been wrong and both were hand-made; a list is checkable
  against the tree and a number is not.
- The missing `uv build` proof: taken, and it was the right read - R1.1 lived
  exactly in the gap. The DoD gains "the published artifact set installs
  clean", proven by rehearsing the release job's own sequence locally in both
  directions (root-only build fails to install; `--all-packages` installs and
  reports `scufris 0.1.0`). The durable half is `check_agreement`'s member arm,
  which fails in CI rather than needing the rehearsal to be remembered.

Verification notes:

- `nix flake check` failed on its first run here, on
  `tests/test_app.py::test_orchestrator_chat_uses_server_cwd`
  (`assert None == 'mock-session'`). That is the known order-dependent failure
  already tracked by 20260803-043935 and 20260803-100411, not a regression from
  this diff: the test passes in isolation, and a second `nix flake check` on the
  same tree returned "all checks passed!" (rc 0). `nix build .#scufris`
  succeeded on the first run.
- Independently re-derived: the domain-free helper genuinely bites. On a `/tmp`
  copy of `scufris_core`, a `rows.py` declaring one table fires all three arms
  (allowlist, `__tablename__`, declarative class); the untouched tree returns
  `[]`; and the pre-move tree reports thirteen `__tablename__` hits.
- Proofs run: `python -c "import scufris_core, scufris_core.engine"` in a dev
  shell rebuilt after `uv lock`; `python -m pytest` (1112 passed, 1 skipped,
  counted from the progress output - matches the close-out claim);
  `tests/test_package_boundaries.py` + `tests/test_examples.py` (4 passed);
  `python -m pytest tests/test_examples.py -k core` (1 passed);
  `python scripts/check_file_size.py` clean with `"packages"` present;
  `rg -q 'packages/\*/tests' pyproject.toml`;
  `test -f tasks/20260803-213242/DECISION.md`; `ruff check`,
  `ruff format --check`, `mypy` clean over 234 files.
- Exactly one `DeclarativeBase` subclass exists in the tree
  (`packages/core/src/scufris_core/base.py:18`), so the Alembic metadata stays
  single; `scufris/db/models.py` imports it and `migrations/env.py` needed no
  edit, as the plan said.

Pending user checks:

- `manual:` in the parent epic 20260803-213242 (Manual Acceptance, line 188) -
  the maintainer names the owning package for a given concern. Still
  `(pending)`, correctly not self-ticked. It does not block this verdict.

## Round 2

- REVIEWER: out-of-context
- VERDICT: APPROVE

- [ ] R2.1 (NIT) docs/RELEASING.md:25 - step 1 tells the operator to bump
  `pyproject.toml` AND every `packages/*/pyproject.toml`, but never says to
  re-run `uv lock`, while `AGENTS.md:71` - written by this same diff - makes a
  lock refresh mandatory after any pyproject edit. `uv.lock:1290` carries
  `version = "0.1.0"` for `scufris-core`, so a bumped-but-unlocked tree passes
  `python -m scripts.release_tools check` and still hands the nix build the old
  version. Neither `docs/RELEASING.md` nor `scripts/check-release-ready.sh`
  mentions `uv lock` anywhere else (verified by grep), so nothing downstream
  catches it. Add "then run `uv lock` and commit the result" to step 1.

  Response:

All twelve round-1 findings are confirmed fixed and ticked. The out-of-context
reviewer verified each Response against the tree; the notable ones:

- R1.1 - `uv build --all-packages` at `release.yaml:239` and the
  one-resolution `dist/*-py3-none-any.whl` install at `:260`, re-derived in
  both directions: the fixed sequence installs and reports `scufris 0.1.0`;
  a root-only build plus its single wheel still fails "requirements are
  unsatisfiable".
- R1.1 PUSHBACK on `scufris-core==0.1.0` - ACCEPTED. The release attaches
  wheels to a GitHub release and publishes to no registry, so
  `Requires-Dist: scufris-core` resolves exactly when the member wheel is
  installed alongside, which the fixed smoke step proves. The guard job
  (`release.yaml:118` -> `check-release-ready.sh` -> `release_tools check`)
  now fails on any member whose version differs from the root, so the
  unpinned dependency always resolves within the attached set. A literal
  `==0.1.0` would have added a fourth unchecked version string. The residual
  risk - the root wheel installed beside a stale member wheel from an earlier
  release - is real but sits outside the release path and is smaller than the
  pin's maintenance hazard.
- R1.12 - `member_pyprojects` and `check_agreement(member_texts=...)` are
  wired into `main()`, and both new tests fail hard when the fix is reverted.
- R1.2, R1.5, R1.11 - the counts in the record were re-derived against the
  tree rather than re-read: eleven `logsetup` importers, twelve
  `scufris_core/engine.py` sites in eleven files, eight pre-existing task IDs
  under `examples/`. All match.

Process signal:

- The census response is applied rather than acknowledged: the `logsetup`
  Step now pastes the eleven names. Both re-derived counts matched, which is
  the first time in this task's history that a hand-made number survived
  checking - because it stopped being hand-made.
- The `uv build` gap is closed durably rather than by rehearsal. The member
  arm of `check_agreement` fails in CI, so the proof does not depend on
  anyone remembering the local sequence.

Verification notes:

- Re-derived independently of the reviewer, in a dev shell rebuilt after
  `uv lock`: `python -c "import scufris_core, scufris_core.engine"` resolves
  to `packages/core/src/scufris_core/__init__.py`; `python -m pytest` rc 0
  with no FAILURES block (the order-dependent `test_app.py` failure tracked by
  20260803-043935 did not appear); `ruff check` and `ruff format --check`
  clean over 234 files; `mypy` clean over 219; `python scripts/check_file_size.py`;
  `python -m scripts.release_tools check` ("version sources agree on 0.1.0");
  both `rg` DoD proofs; `test -f tasks/20260803-213242/DECISION.md`;
  `pytest tests/test_examples.py -k core` rc 0.
- `nix flake check` -> "all checks passed!" and `nix build .#scufris` -> rc 0,
  both on the first run this round. Neither was in the reviewer's set, so this
  closes DoD proof 8 in-session.
- R2.1 re-derived rather than taken: `grep -n 'uv lock\|uv.lock' docs/RELEASING.md
  scripts/check-release-ready.sh` returns nothing, and `uv.lock:1290` does
  carry the member version.

Pending user checks:

- `manual:` in the parent epic 20260803-213242 (Manual Acceptance, line 188) -
  the maintainer names the owning package for a given concern. Still
  `(pending)`, correctly not self-ticked. It does not block this verdict.
