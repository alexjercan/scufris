# Move read-only host inspection into packages/host

- PRIORITY: 104
- TAGS: refactor, v0.2.0, architecture, host
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260803-213242

## Story

As the Scufris maintainer, I want read-only host inspection moved into
`packages/host`, so that the one product surface surviving the rewrite
untouched - Stats - sits behind a package that needs no privilege and imports no
sibling but `core`.

No behavior changes. This is a move.

The move's real work is not `git mv`. `scufris/` reaches into the host tree's
INTERNALS in thirteen places - `hostd/engine.py:33`, `hostd/preview.py:26-29`,
`hostd/nixos.py:41-43`, `hostd/executor.py:25`, `hostd/actions/plans.py:12-13`,
`hostd/actions/validate.py:18`, `hostconfig/resolve.py:17`,
`hostconfig/changes.py:23`, `app.py:57`, `api/host.py:33` - and once the tree is
a sibling distribution every one of them violates the epic's one rule.
`test_no_package_imports_a_sibling_private_module` is written and green today
precisely because `core` gave it nothing to police; its own docstring says it
"earns a red run once a second package is carved out beside `core`". This task
is that run. Routing those thirteen through the facade is the task.

## Steps

- [x] Create `packages/host/` with its own `pyproject.toml` declaring ONLY
      `psutil` and `pydantic`. NOT `scufris-core`: confirmed by inspection, the
      whole moving tree (`scufris/host/*.py`, `metrics.py`, `processes.py`)
      imports nothing but stdlib, `psutil` and `pydantic` - no `scufris`
      module, not even `logsetup`, and no database. The epic declares
      `host -> nothing`; this package is the proof the rule is real. Mirror
      `packages/core/pyproject.toml`: same `requires-python`, hatchling,
      `packages = ["src/scufris_host"]`, `version = "0.1.0"` (the root's - it is
      what `release_tools.check_agreement` asserts and
      `tests/test_release.py:94` checks).
- [x] `git mv scufris/host packages/host/src/scufris_host`, with
      `scufris/host/README.md` inside it. Move `scufris/metrics.py` and
      `scufris/processes.py` to `packages/host/src/scufris_host/` beside it -
      both qualify by the same inspection, and `HostStats` is what Stats serves.
      Move `scufris/host/__init__.py`'s `HostInspector`/`HostOverview` unchanged.
- [x] Widen `scufris_host/__init__.py` to cover every name `scufris/` needs, so
      no root module has to reach past it. The existing `__all__` already
      exports all but four; ADD `nix_cli` (from `run`, wanted by six `hostd` and
      `hostconfig` modules), `MIN_HOST_OVERVIEW_TTL` and `HostOverviewCache`
      (from `overview`, wanted by `app.py:57` and `api/host.py:33`), and
      re-export `metrics`' and `processes`' public names. Verified: the three
      name sets are pairwise disjoint - `host.__all__` vs metrics' fifteen vs
      processes' six - so the flat facade needs no aliasing.
- [x] Rewrite every `scufris/` import of the moved tree to the facade. Deep
      imports that MUST become facade imports (the thirteen above). Already
      shallow, so a pure rename: `checks.py:46,207`, `host_watch.py:26`,
      `app.py:56,81,86`, `api/host.py:32,53,54`, `mcp_host_tools/inspection.py`
      (fifteen function-local imports of `..host`), `telegram/wiring.py:42`,
      `telegram/render.py:38`, `telegram/contracts.py:19`. `scufris/digest.py`
      does not import the tree at all; leave it alone. The telegram three make
      the `telegram -> host` edge the epic's graph declares explicit.
- [x] Wire the workspace: add `scufris-host` to the root `dependencies` and
      `[tool.uv.sources]`, add `scufris_host` to
      `[tool.ruff.lint.isort] known-first-party`, DROP `psutil` from the root
      `dependencies` (after the move zero root modules import it; `types-psutil`
      stays in the dev group for the package's own type checking), and
      regenerate `uv.lock`. `[tool.uv.workspace] members = ["packages/*"]` and
      `testpaths = ["tests", "packages/*/tests"]` already cover the new member -
      no change needed, and that is the point of how they were written.
- [x] Move the five test modules that test the moved tree to
      `packages/host/tests/`: `test_host_inspection.py`, `test_host_nix_store.py`,
      `test_host_thermal.py`, `test_metrics.py`, `test_processes.py`. Verified
      they use no fixture from `tests/conftest.py`, so the new directory needs no
      `conftest.py`. `test_host_nix_store.py` does
      `from test_host_inspection import ok`; that import survives only because
      both land in the same directory. Everything else named `test_host_*` tests
      root modules (`host_actions`, `hostd`, the MCP server) and STAYS.
- [x] Re-point the root tests that import the moved names: ~20 modules under
      `tests/`, including `conftest.py:41` (whose `make_fixture_stats()` builds a
      `HostStats`) and `test_host_mcp_server.py`'s function-local
      `from scufris.host import ...`. Root tests may import
      `scufris_host` deeply - `_import_roots()` globs `packages/*/src/*` plus
      `scufris/`, not `tests/` - but use the facade anyway.
- [x] Add `examples/host_report_fixture.py`, an OFFLINE example rendering every
      report from canned fixtures, and add it to `tests/test_examples.py`'s
      `OFFLINE` tuple (today `("core_unit_of_work.py",)`). The seam already
      exists and is documented as existing for this: `FakeRunner` +
      `ok_result` live in `run.py` "so the example script ... can drive an
      inspection without touching the real host", and `HostInspector.__init__`
      takes `config_repo`, `system` and `cpu_sysfs` as `Path`s, so the sysfs- and
      filesystem-backed reports point at `tmp_path`. Cover the seventeen
      `render_*` functions in `render.py`. Scope OUT `HostStats`: `metrics.py`
      calls `psutil` at module level and shells to `nvidia-smi` via `subprocess`
      directly, so it has no injectable seam and adding one is the behavior
      change this task forbids. Leave `examples/host_inspect.py` and
      `examples/nixos_change.py` unlisted - they need a real NixOS box - but do
      re-point their imports.
- [x] Write `test_stats_endpoint_matches_inspector_output` in `tests/`. It does
      not exist; the nearest coverage is `tests/test_app.py` and
      `tests/test_route_contract.py`. Cheap: assert the `/api/stats` body equals
      the fake `Collector`'s `collect()` payload. It is the one assertion that
      makes "Stats still serves the same payload" falsifiable across the move.
      It belongs in `tests/`, not the package: it exercises the composition
      root's route, not the package.
- [x] Update the prose and non-Python references naming the moving tree:
      `README.md:31,135`, `AGENTS.md:19,127,128` (the line numbers in the brief
      were stale), `nix/tests/scufris-vm.nix:66`, `web/src/stats-types.ts:103`,
      and add a CHANGELOG entry under `[Unreleased]` as the `core` carve did.
      `flake.nix` needs NO change: `mkCheckWith` already exports `REPO_ROOT`
      for editable members.

## Definition of Done

- The package imports on its own and depends on NO sibling, `core` included
  (cmd: `uv run python -c "import scufris_host"`;
  test: `test_no_package_imports_a_sibling_private_module`).
- Its own suite passes unmoved in behavior AND still runs in the canonical gate
  (cmd: `python -m pytest packages/host/tests && python -m pytest --collect-only | rg -q packages/host`).
- The offline example renders every report from fixtures with no host access
  (cmd: `python -m pytest tests/test_examples.py -k host_report_fixture`).
- Stats still serves the same payload
  (test: `test_stats_endpoint_matches_inspector_output`).
- The whole gate is green
  (cmd: `python -m pytest && ruff check . && mypy scufris packages`).

## Notes

- Parent: 20260803-213242.
- This package needs NO privilege. If something here needs root, it belongs in
  `hostctl` or `hostd`.
- Stats is the one page that survives the UI demolition intact. Keep it working
  at every commit.
- Proofs run RED on the base (`e7cb027`), in `nix develop`: `import
  scufris_host` -> 1; `pytest packages/host/tests` -> 4 (no such path);
  `--collect-only | rg packages/host` -> 1; `pytest tests/test_examples.py -k
  host_report_fixture` -> 5; `-k test_stats_endpoint_matches_inspector_output`
  -> 5. `tests/test_package_boundaries.py` and the five moving test modules are
  green on the base, which is the no-behavior-change baseline.
- Run verification through `nix develop --command`. A bare `uv run` targets
  `.venv`, which has no pytest; the dev shell is the environment the flake
  checks use.
- The facade-widening decision is recorded in `DECISION.md`.

## Close-out

**What and why.** `scufris/host/`, `scufris/metrics.py` and
`scufris/processes.py` are now the `scufris-host` distribution under
`packages/host`, import root `scufris_host`, declaring `psutil` and `pydantic`
and no sibling. Every `scufris/` import of the tree goes through the facade, so
`test_no_package_imports_a_sibling_private_module` polices a real pair for the
first time. No behavior changed: the same endpoints, the same payloads, the same
wheel.

**Three plan corrections, all forced by the code.**

1. **`HostInspector`/`HostOverview` moved to a new `inspector.py`.** The plan
   said re-export `overview`'s names from `__init__` and leave the two classes
   where they were; those are mutually exclusive. `overview.py` does
   `from . import HostInspector, HostOverview`, and both classes are defined
   BELOW the import block in `__init__.py`, so `from .overview import ...` in
   the facade is a cycle that fails at import. Three ways out: put the import at
   the bottom of `__init__` (ruff E402, and a `noqa` guarding a cycle), guard
   `overview`'s import with `TYPE_CHECKING` (works - every use is an annotation
   under `from __future__ import annotations` - but leaves a subtle constraint
   for the next editor), or move the classes to their own module. The third was
   chosen: `__init__` is now purely re-exports, which is the shape the epic
   wants of a distribution root, and `hostd`/`hostctl` inherit a door that
   cannot re-develop the cycle when they carve.
2. **`NIX_FEATURES` had to join the facade.** The plan enumerated three names to
   add. `tests/test_host_actions.py` also imports `NIX_FEATURES` from `run`, so
   `NIX_FEATURES` makes four.
3. **`tests/fixtures/host/` moved with the five test modules.** The plan did not
   mention it. Six files of output captured from the real host, used by
   `test_host_inspection.py` and `test_host_nix_store.py` and by nothing else,
   so it moved to `packages/host/tests/fixtures/host/`.

Two smaller corrections: `render.py` has fourteen `render_*` functions plus
`human_bytes`, not "seventeen" as the plan said; and `/api/stats` calls
`collector.sample()`, not `collect()`.

**Difficulties.** `nix develop` is built FROM `uv.lock`, so regenerating the
lock with `nix develop --command uv lock` cannot work - the shell fails to
evaluate with `attribute 'scufris-host' missing` before the command runs. Ran
`uv lock` with the `uv` already on PATH, then re-entered. The flake also
evaluates the git tree, so `packages/host/pyproject.toml` had to be `git add`ed
before `nix develop` could see the new member.

**Evidence.**

- `import scufris_host` -> 0, 89 names in `__all__` (90 before round 4 dropped
  `DEFAULT_CONFIG_REPO`).
- `pytest` -> 1117 passed, 1 skipped. `packages/host/tests` -> 55 passed and 55
  collected under the bare `pytest --collect-only`.
- `ruff check .`, `ruff format --check .`, `mypy scufris packages` all clean.
- `nix flake check` -> all 5 checks passed. `nix build .#scufris .#scufris-web`
  builds, including the new `scufris-host-0.1.0` derivation.
- Both new proofs were falsified before being trusted:
  `test_no_package_imports_a_sibling_private_module` goes RED when a single
  `from scufris_host.run import Runner` is added to `scufris/checks.py`, and
  `test_stats_endpoint_matches_inspector_output` goes RED when the expected
  payload's `hostname` is changed.

**Reflection.** The plan's inspection work paid off - the thirteen call sites
and the disjoint name sets were all exactly as recorded, so the rewrite was
mechanical. What it could not see from a static read was the import CYCLE that
widening the facade creates; that only appears when you try it. A plan step that
says "re-export X from `__init__`" is worth checking against whether X's module
imports `__init__` back.

## Review round 1

Five of six findings fixed - four doc surfaces (`AGENTS.md`'s `nix_cli` rule,
`nix/scufris-service.nix`'s tool table, `docs/RELEASING.md`'s `Requires-Dist`
list, and correction 2 above) plus one real defect in
`examples/host_report_fixture.py`, whose `sys.path` insert pointed at the repo
root and so only ever worked because the dev shell has the package installed.
R1.6 got reasoned pushback rather than a fix: `tests/test_app.py` is on the
ratchet on the base and its split belongs to 20260729-103712.

Two things the round taught. First, the example proof cannot see a broken
`sys.path` insert, because the environment it runs in supplies the import
anyway - the venv ships `_editable_impl_scufris_host.pth`, whose one line is
`<worktree>/packages/host/src`. `-P` does not isolate from that: it drops cwd
and the script dir but leaves `site` to process the `.pth`, so under `-P` alone
the broken repo-root insert passes too, and the first attempt at falsifying it
proved nothing. `-S` bypasses `site` - but it also drops site-packages
wholesale, so the example then dies on `pydantic` instead. The method that
isolates the `.pth` and keeps the deps is `-S -P` from outside the tree with
`PYTHONPATH` pointed at the venv's `site-packages`, since `PYTHONPATH` entries
get no `.pth` processing. Under it the old insert fails and the current one
renders. A proof that passes for a reason other than the one it claims is worth
as little as a proof that fails - and so is the flag you reach for to check it.
Second, the reviewer caught a
miscount inside a correction to a miscount, which is the same failure mode
twice: counting call sites and names by hand instead of deriving them. The
count that was right (thirteen call sites) came from a grep.

One unrelated red appeared during re-verification:
`tests/test_app.py::test_orchestrator_chat_uses_server_cwd` failed one
full-suite run, passed alone, and passed the next full run on the same tree.
Pre-existing and untouched by this branch, so it is filed as 20260804-003731
rather than folded in.

The `import scufris_host` proof reads RED in a worktree whose `.venv` predates
the new member; `uv sync` in the worktree fixes it. Environment staleness, not
a code state - worth knowing before treating that exit 1 as a regression.

## Review rounds 2 and 3

Both rounds found exactly one thing each, and both times it was a sentence
about how something was checked rather than the thing itself. The code has not
changed since round 1's fix commit `b91a8d0`.

R2.1: the R1.3 Response claimed `python -P` isolated the example from the dev
shell's install. It does not. `-P` drops cwd and the script directory but
leaves `site` running, and `site` processes
`_editable_impl_scufris_host.pth`, whose one line is
`<worktree>/packages/host/src` - so `scufris_host` imported either way and the
run distinguished nothing. The old repo-root insert had never been falsified.

R3.1: the correction's own first draft, and round 2's recording prose, then
reached for `-S -P`. That bypasses `site` and the `.pth`, but `-S` drops
site-packages wholesale, so the example dies on `pydantic` before rendering
anything. A method that cannot run the script cannot confirm the script.

The method that actually isolates: `-S -P` from outside the tree with
`PYTHONPATH` pointed at the venv's `site-packages`. `-P` removes cwd, `-S`
leaves the `.pth` unprocessed, and the explicit `PYTHONPATH` restores
`pydantic` and `psutil` without restoring the `.pth`, because path entries from
`PYTHONPATH` get no `.pth` processing at all. Derived that last point on a
scratch directory rather than assuming it: a `.pth` there is ignored when the
directory arrives via `PYTHONPATH`, and honoured when the same directory is
passed to `site.addsitedir`. Under the working method the bare import and the
old repo-root insert both raise `ModuleNotFoundError`, and the current insert
exits 0 rendering 140 lines.

**Reflection.** Three rounds, three findings, and all three were counts or
methods recalled from memory instead of transcribed from a run - the miscount
in round 1, the implementer's evidence line in round 2, the reviewer's own
recording prose in round 3. The pattern is not carelessness about the work; the
work was right each time. It is that a "verified by" sentence gets written as a
summary of intent after the fact, and nothing checks it, because it sits in
prose rather than in a rig. The cheap fix is mechanical: paste the command and
its exit code, and let the sentence be a caption on evidence rather than a
recollection of it. The expensive lesson underneath is that an environment
which makes an import succeed for a second reason - here an editable install -
turns every naive isolation check green, so isolation has to be proven by a
control that FAILS, not only by a run that passes.

## Review round 4

The first round since round 1 to touch code, and the first to reach surfaces
outside the two records the earlier rounds circled.

R4.3 dropped `DEFAULT_CONFIG_REPO` from the facade. It was defined at
`inspector.py:58`, referenced once at `inspector.py:95` by the package itself,
exported by no `master` facade and wanted by no consumer - `rg` across the
worktree finds it in `__init__.py` and `inspector.py` and nowhere else. An
unrequested export in the one namespace this task exists to keep deliberate,
so it went. `__all__` is 89 names, not 90.

R4.1, R4.2 and R4.4 were the same defect in three surfaces: a number or a
module path written from memory. All three fixes were derived by a rig rather
than recounted.

- `__all__`: `master`'s 63 entries against the branch's 90 (before R4.3) is a
  delta of 27; the `from .metrics import` block carries 16 names and
  `from .processes import` 6, so 22 re-exports leave 5 added names, and R4.3
  takes it to four. DECISION.md said "four plus twenty-one" - right total,
  wrong split, and it now carries the derivation beside the number. Its
  decision 1 also listed three added names while claiming four, and its
  decision 2 said metrics has fifteen public names when an AST walk finds
  sixteen; both corrected in the same pass, since a document the later
  `hostd` and `hostctl` carves read cannot carry a count no rig produced.
- `render_*`: `rg -c '^def render_'` -> 14. The facade docstring said fifteen,
  which the close-out had already corrected for the plan and missed here.
- `tests/test_host_actions.py:51` named `host.run.nix_cli` - a module path that
  no longer exists, in the exact deep form
  `test_no_package_imports_a_sibling_private_module` now forbids, five lines
  below an import this diff had already rewritten.

Round 4's process signals stand as recorded, and one has no fix: commit
`b91a8d0`'s message still carries the `python -P` claim round 2 disproved.
History is immutable; the correction lives in the records instead.

**Reflection.** Four rounds, and the four findings that were counts all came
from the same habit - a number typed while writing prose, next to code that had
already moved past it. What broke the pattern here was not more care but a
different input: an AST walk over both revisions of `__all__` and two `rg -c`
runs, pasted into the record as the derivation rather than summarised as a
conclusion. The doc sweep also has a blind spot worth naming: it greps for
identifiers, so it catches `host.run.nix_cli` but never catches "fifteen". A
count is a claim about code with no token in common with the code it describes,
so nothing mechanical will find it stale. The only defence is to not write one
without the command that produced it.
