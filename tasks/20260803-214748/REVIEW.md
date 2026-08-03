# Review: Move read-only host inspection into packages/host

- TASK: 20260803-214748
- BRANCH: refactor/host-package

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

The move itself is sound and the gate is green: `python -m pytest` exit 0,
`ruff check .`, `ruff format --check .` and `mypy scufris packages` (154 files)
all clean, re-run by the recording pass. All five proofs pass on their stated
criterion; `test_package_boundaries.py` genuinely bites on a deep
`scufris_host.run` import. Every finding below is a doc-sweep or bookkeeping
miss, not a defect in the carve.

- [x] R1.1 (MAJOR) AGENTS.md:136 - the diff updated three lines of this file
  for the new layout (18, 19, 127) but left the rule
  "route new CLI calls through `host.run.nix_cli`" pointing at a module that no
  longer exists, and at a DEEP import that the boundary this task exists to
  enforce now forbids. Rewrite the line as "route new CLI calls through
  `scufris_host`'s `nix_cli`", naming the facade, not `run`.
  - Response: fixed, verbatim. The line was the one place in a live doc that
    still instructed a future editor to write the exact deep import
    `test_no_package_imports_a_sibling_private_module` now fails on.
- [x] R1.2 (MINOR) nix/scufris-service.nix:141 - line 138 was renamed to
  `scufris_host`, but the four table rows directly below it still cite
  `host/units.py`, `host/journal.py`, `host/packages.py`, `host/storage.py` and
  `host/network.py`, none of which exist at those paths. Rewrite the rows as
  `scufris_host/units.py`, `scufris_host/journal.py`,
  `scufris_host/packages.py`, `scufris_host/storage.py`,
  `scufris_host/network.py`.
  - Response: fixed. All four rows renamed. The finding names five modules
    across four rows, which is the count; `packages.py` appears twice.
- [x] R1.3 (MINOR) examples/host_report_fixture.py:40 - the insert under
  "Run from a checkout without installing it" adds the REPO ROOT, but the
  script's only import is `scufris_host`, which lives at
  `packages/host/src/scufris_host`; the checkout case the comment promises
  still fails, and the example passes its proof only because the dev shell has
  the package installed. Insert
  `Path(__file__).resolve().parent.parent / "packages" / "host" / "src"`,
  matching `examples/core_unit_of_work.py:32`.
  - Response: fixed, and the finding was right that the proof could not see it.
    The insert now points at `packages/host/src`, with the same two-line
    comment `core_unit_of_work.py` carries. Falsified the old path and
    confirmed the new one by running the script from `/tmp` under
    `python -S -P` with `PYTHONPATH` set to the venv's `site-packages`: `-P`
    drops cwd and the script dir, `-S` skips `site` and so leaves
    `_editable_impl_scufris_host.pth` unprocessed, and the explicit
    `PYTHONPATH` restores `pydantic` and `psutil` without restoring that
    `.pth` (path entries from `PYTHONPATH` do not get `.pth` processing). Under
    that isolation a bare `import scufris_host` raises `ModuleNotFoundError`,
    the same import with the OLD repo-root insert also raises it, and the
    script with its current insert exits 0 rendering 140 lines of real report
    output. (This sentence first claimed `python -P` alone; see R2.1 - that
    does not disable `.pth` processing, so the run it recorded distinguished
    nothing.) Left
    the `tests/test_examples.py` proof as-is: it runs examples the way a
    developer in the shell does, and adding an isolation flag there would be
    this task inventing an example contract the core carve did not set.
- [x] R1.4 (MINOR) tasks/20260803-214748/TASK.md:173 - close-out correction 2
  says "the plan enumerated four names to add ... so the count is five", but
  the facade diff adds exactly four (`HostOverviewCache`,
  `MIN_HOST_OVERVIEW_TTL`, `NIX_FEATURES`, `nix_cli`), which is what
  `DECISION.md` records. The plan ENUMERATED three. Restate as "the plan
  enumerated three; `NIX_FEATURES` makes four", so the record matches the code.
  - Response: fixed. Re-derived rather than accepted: Step 3 of the plan names
    `nix_cli`, `MIN_HOST_OVERVIEW_TTL` and `HostOverviewCache` - three - and
    the facade diff adds those plus `NIX_FEATURES` - four. The correction that
    was supposed to fix a miscount had itself miscounted.
- [x] R1.5 (NIT) docs/RELEASING.md:7 - "the root wheel's
  `Requires-Dist: scufris-core`" now understates the set; the root also
  declares `Requires-Dist: scufris-host`. Change to
  "`Requires-Dist: scufris-core` and `scufris-host`".
  - Response: fixed, with the verb agreed to the new plural.
- [x] R1.6 (NIT) tests/test_app.py:249 - the new
  `test_stats_endpoint_matches_inspector_output` grows a 4270-line file that
  only passes the `filesize` ratchet via `ALLOWLIST`. Consider landing it in a
  new `tests/test_stats_contract.py` instead.
  - Response: not fixed, and deliberately not. `tests/test_app.py` was already
    on the ratchet on the base (`master`'s `ALLOWLIST` is exactly
    `{"tests/test_app.py"}`), so this diff neither adds an entry nor changes
    the ratchet's state; it adds 30 lines to a file already over cap. That
    file's split is owned by 20260729-103712, and 20260731-171432's notes say
    so in as many words: "`tests/test_app.py` (3813) and `scufris/app.py` stay
    allowlisted and belong to 20260729-103712. Do not touch either." Splitting
    a stats module out here would take a bite of that task's scope from a
    refactor whose whole contract is no behavior change. The test also reads
    against `test_api_stats_returns_snapshot` directly above it - its docstring
    explains what that spot-check would miss - and uses `_settings`,
    `fake_collector` and `fake_stats` from this module. Moving it costs that
    adjacency and duplicates the rig to save a file the repo has already
    decided someone else will split.

Process signal: the plan's counts were off in three places - fourteen renderers
not seventeen, `sample()` not `collect()`, and the facade name count. The
close-out corrects the first two honestly and miscorrects the third. Counting
call sites by hand in a plan is what produced all three.

Verified by the recording pass, independently of the out-of-context reviewer:
`ruff check .`, `ruff format --check .`, `mypy scufris packages`,
`python -m pytest` (exit 0), proofs 2, 3 and 4 by exit code,
`tests/test_package_boundaries.py`, and `scufris_host.__all__` against the
facade diff (the re-derivation behind R1.4). R1.1 and R1.2 were re-derived from
the working tree, not accepted on the reviewer's word. Not verified:
`nix build .#scufris .#scufris-web` and the KVM-only VM tests. No `manual:`
proofs are outstanding.

## Round 2

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

All six round-1 findings are resolved in the code and the docs: R1.1 through
R1.5 confirmed fixed by re-derivation from the tree, and R1.6's pushback
confirmed sound - `master`'s `ALLOWLIST` is exactly `{"tests/test_app.py"}`,
unchanged at HEAD, so the diff adds lines to an already-allowlisted file
without touching the ratchet, and `tasks/20260731-171432/TASK.md:183-184`
carries the "Do not touch either" instruction verbatim. Boxes ticked above.
The fix commit b91a8d0 introduces no regression: four doc/example surfaces
plus records, comment-only in the `.nix`.

The one finding is the R1.3 fix's EVIDENCE, not the fix. The `sys.path` insert
is correct - re-derived by running the example from `/tmp` under
`python -S -P` with `PYTHONPATH` set to the venv's `site-packages`, where
`scufris_host` resolves from `packages/host/src` through the script's own
insert. (Under `-S -P` alone, as this sentence originally read, the script does
not run at all: `-S` drops site-packages and it dies on `pydantic`. See R3.1.)

- [x] R2.1 (MAJOR) tasks/20260803-214748/REVIEW.md:45 - the R1.3 Response
  claims the fix was confirmed and the old path falsified "by running the
  script under `python -P` with `PYTHONPATH` cleared, from `/tmp`: that
  isolates it from cwd and from the dev shell's installed package". It does
  not isolate it from the installed package. The venv ships
  `_editable_impl_scufris_host.pth`, whose single line is
  `<worktree>/packages/host/src`; `.pth` files are processed by `site`, which
  `-P` does not disable, so `cd /tmp && env -u PYTHONPATH .venv/bin/python -P
  -c "import scufris_host"` exits 0 with the module resolved from that `.pth`.
  Under that method the OLD repo-root insert would have passed identically, so
  the run recorded cannot have distinguished the two paths and the old path was
  never falsified. Replace the sentence with the method that does isolate -
  `python -S -P` from outside the tree, which bypasses `site` and therefore the
  `.pth` - or restate it as "confirmed by inspection against
  `core_unit_of_work.py`; the dev shell's editable install makes the import
  succeed either way". Make the same correction at
  `tasks/20260803-214748/TASK.md:222-225`, which repeats the claim and builds
  its lesson on it.
  - Response: fixed, and re-derived rather than accepted. Confirmed the venv's
    `_editable_impl_scufris_host.pth` holds exactly
    `<worktree>/packages/host/src`, then ran both interpreters from `/tmp` with
    `PYTHONPATH` unset: `-P` alone imports `scufris_host` and resolves it to
    `packages/host/src/scufris_host/__init__.py` (exit 0), while `-S -P` raises
    `ModuleNotFoundError` (exit 1). The finding is right on both halves - `-P`
    never isolated from the editable install, so the old repo-root insert would
    have passed identically and the recorded run falsified nothing.
    One correction to the finding's first offered rewrite: `python -S -P`
    alone does not run the EXAMPLE, because `-S` drops site-packages
    wholesale and the script dies on `from pydantic import BaseModel`. The
    method that isolates the `.pth` while keeping the deps is `-S -P` with
    `PYTHONPATH` pointed at the venv's `site-packages` - `PYTHONPATH` entries
    get no `.pth` processing, so `pydantic` and `psutil` come back and
    `scufris_host` does not. Under it: bare import -> `ModuleNotFoundError`;
    old repo-root insert -> `ModuleNotFoundError`; current insert -> exit 0,
    140 lines of report output. Both surfaces the finding names -
    `REVIEW.md`'s R1.3 Response and `TASK.md`'s `## Review round 1` paragraph -
    now carry that method, say why each flag is load-bearing, and mark the
    original `-P` claim as wrong rather than silently overwriting it, so the
    correction stays visible in the record.

Process signal: two rounds running, the record's weak point is proof-method
claims rather than code. R1.3's fix was right and its evidence sentence wrong,
which is the same shape as round 1's miscount inside a correction to a
miscount. Both times the code was sound and the sentence about how it was
checked was not.

Verified by the recording pass, independently of the out-of-context reviewer:
`ruff check .`, `ruff format --check .`, `mypy scufris packages` (154 files),
`.venv/bin/python -m pytest` (exit 0), the `.pth` contents and the two
interpreter runs behind R2.1, and the example's import resolution from `/tmp`
under `-S -P` with `PYTHONPATH` set to the venv's `site-packages` (this
qualifier added per R3.1; `-S -P` alone cannot run the example).
Not verified: `nix flake check`,
`nix build .#scufris .#scufris-web`, the KVM-only VM tests. No `manual:` proofs
are outstanding.

## Round 3

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

R2.1 is resolved and its box is ticked. Every checkable claim in its Response
reproduced, derived rather than accepted: the `.pth` holds exactly
`<worktree>/packages/host/src`; `-P` alone from `/tmp` with `PYTHONPATH` unset
imports `scufris_host` and resolves it to
`packages/host/src/scufris_host/__init__.py` at exit 0, so the old repo-root
insert would indeed have passed identically; `-S -P` alone raises
`ModuleNotFoundError` and cannot run the example at all, dying on `pydantic`;
and under `-S -P` with `PYTHONPATH` set to the venv's `site-packages` the bare
import and the old insert both raise `ModuleNotFoundError` while the current
insert exits 0 rendering 140 lines. The Response's load-bearing general claim -
that `PYTHONPATH` entries get no `.pth` processing - was re-derived from
scratch on a scratch directory: a `.pth` there is ignored via `PYTHONPATH` with
`site` enabled, and picked up by `site.addsitedir` on the same directory. The
Response also pushes back correctly on the finding's first offered rewrite
instead of accepting it, and marks the original claim as wrong in place.

The round-2 commits introduce no regression: `git diff 2e051a4..HEAD` touches
only `tasks/20260803-214748/{REVIEW,TASK}.md`. The last code commit is round
1's b91a8d0.

The one finding is round 2's OWN recording prose, carrying the third instance
of this branch's recurring defect.

- [x] R3.1 (MAJOR) tasks/20260803-214748/REVIEW.md:125 - round 2's body says
  the insert was "re-derived by running the example under `python -S -P` from
  `/tmp` with `PYTHONPATH` cleared, where `scufris_host` resolves from
  `packages/host/src` through the script's own insert". That run cannot have
  happened: under `-S -P` the example dies with
  `ModuleNotFoundError: No module named 'pydantic'` before rendering anything,
  because `-S` drops site-packages wholesale - which is the fact R2.1's own
  Response records twenty lines below, so the record now contradicts itself
  about how the example was checked. Graded MAJOR to match R2.1, which is the
  identical defect one round earlier: a stated verification method that does
  not do what the sentence claims. Restate line 125 as "re-derived by running
  the example from `/tmp` under `python -S -P` with `PYTHONPATH` set to the
  venv's `site-packages`; under `-S -P` alone the script does not run at all,
  dying on `pydantic`", and qualify the same claim at line 178, which repeats
  it as "the example's import resolution under `-S -P` from `/tmp`".
  - Response: fixed, and re-derived rather than accepted. Ran the example from
    `/tmp` under `-S -P` with `PYTHONPATH` unset: it exits 1 on
    `ModuleNotFoundError: No module named 'pydantic'`, raised from
    `scufris_host/inspector.py:14` by way of the facade, so no report is
    rendered and the sentence at line 125 describes a run that cannot have
    produced what it claims. Both spots restated with the `PYTHONPATH`
    qualifier, and both mark the original wording as wrong rather than
    overwriting it - the same in-place correction R1.3's Response took, so the
    record keeps its own errors visible.
    Accepting the severity: the finding is right that this is R2.1's defect a
    round later, and it is worse for being in the reviewing side's prose, which
    is where the branch's honesty checks are supposed to originate.

Process signal: three rounds running, and every finding has been a
proof-method sentence rather than code - round 1 a miscount, round 2 the
implementer's evidence line, round 3 the reviewer's own. The constant is that
the method was described from memory after the fact. A "verified by" line
should paste the command actually run, not a recollection of it.

Verified by the recording pass, independently of the out-of-context reviewer:
`ruff check .`, `ruff format --check .`, `mypy scufris packages`,
`python -m pytest` (exit 0) in `nix develop`; all five proofs by exit code on
their own criterion, including `test_no_package_imports_a_sibling_private_module`
and `pytest --collect-only | rg packages/host`; the `.pth` contents; the four
interpreter runs behind R2.1; and the scratch-directory derivation of the
`PYTHONPATH`-versus-`addsitedir` claim. R3.1 was re-derived by running the
example under `-S -P` and reading REVIEW.md:125 and :178 in the tree, not
accepted on the reviewer's word. Not verified: `nix flake check`,
`nix build .#scufris .#scufris-web`, the KVM-only VM tests. No `manual:` proofs
are outstanding.

## Round 4

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

R3.1 is resolved and its box is ticked. Both surfaces the finding named carry
the working method with the `PYTHONPATH` qualifier and mark the original
wording as wrong rather than overwriting it. Re-derived rather than accepted:
the example run from `/tmp` under `-S -P` with `PYTHONPATH` unset exits 1 on
`ModuleNotFoundError: No module named 'pydantic'`, and the same run with
`PYTHONPATH` set to the venv's `site-packages` exits 0 rendering exactly 140
lines. The round-3 commits touch only
`tasks/20260803-214748/{REVIEW,TASK}.md`; the last code commit is still round
1's `b91a8d0`, so no fix commit introduced a regression.

This round closes the three items rounds 1 through 3 all recorded as not
verified. `nix flake check` passes; `nix build .#scufris .#scufris-web` exits
0; all three check derivations (`pytest`, `records`, `filesize`) build in the
sandbox; and the KVM-only `nix build .#scufris-vm-test` builds green on this
machine. Nothing outstanding is left unrun on this branch.

The three findings are the doc-and-record sweep the earlier rounds missed. Two
are the same stale-reference class as R1.1, in surfaces that sweep did not
reach; one is this branch's fourth miscount.

- [x] R4.1 (MAJOR) tasks/20260803-214748/DECISION.md:88 - the Consequences
  bullet records "Four names added to `__all__` plus twenty-one re-exports",
  and both halves are wrong. Derived from the code rather than recounted by
  hand: `master`'s `scufris/host/__init__.py` `__all__` holds 63 entries and
  the new facade's holds 90, a delta of 27; the `from .metrics import` block
  re-exports 16 names and `from .processes import` 6, which is 22 re-exports,
  leaving 5 added names - `nix_cli`, `MIN_HOST_OVERVIEW_TTL`,
  `HostOverviewCache`, `NIX_FEATURES` and `DEFAULT_CONFIG_REPO`. Restate as
  "Five names added to `__all__` plus twenty-two re-exports". Graded MAJOR
  under the honesty dimension - a recorded number no rig produced - and
  because DECISION.md is the durable record the epic's later carves
  (`hostd`, `hostctl`) read, unlike round 1's R1.4 miscount which sat in a
  close-out correction. It is also this branch's fourth count-or-method
  recalled instead of derived.
  - Response: fixed, and re-derived rather than accepted. An AST walk over
    `__all__` on both revisions gives 63 on `master` and 90 on the branch, a
    delta of 27 with nothing removed; the `from .metrics import` block carries
    16 names and `from .processes import` 6, so 22 re-exports leave 5 added
    names - the reviewer's split confirmed. R4.3 then drops
    `DEFAULT_CONFIG_REPO`, so the bullet now reads "Four names added to
    `__all__` plus twenty-two re-exports" and carries the derivation beside it.
    Two more counts in the same document were wrong the same way and are fixed
    in the same pass: decision 1 claimed "but four" while listing three names
    (`NIX_FEATURES` was the unlisted fourth, and it is wanted by
    `tests/test_host_actions.py` and `examples/nixos_change.py`, not by a root
    call site), and decision 2 said metrics has fifteen public names when an
    AST walk finds sixteen.
- [x] R4.2 (MINOR) tests/test_host_actions.py:51 - the comment "see
  `host.run.nix_cli` for why they are there at all" names a module path that
  no longer exists AND names the exact deep form
  `test_no_package_imports_a_sibling_private_module` now forbids. Same defect
  as R1.1, in a file this diff already edits five lines above, so the doc
  sweep had the file open. Rewrite as "see `scufris_host`'s `nix_cli`".
  - Response: fixed, verbatim. Confirmed by `rg` that no other live surface
    outside `tasks/` still names `host.run.`, `host.models.` or
    `host.overview.`; CHANGELOG.md's two `scufris.host` mentions are the entry
    describing this very rename and stay.
- [x] R4.3 (MINOR) packages/host/src/scufris_host/__init__.py:159 -
  `DEFAULT_CONFIG_REPO` is newly added to `__all__`; it is defined at
  `inspector.py:58` and used only at `inspector.py:95`, by the package itself,
  and `master`'s facade did not export it. No Step asks for it and no consumer
  wants it, so it is an unrequested export widening the one namespace this
  task's whole point is to keep deliberate. Drop it from `__all__` and from
  the `.inspector` import line - which also drops R4.1's added-name count from
  five to four, so fix that bullet to whichever pair is true after this
  choice.
  - Response: fixed, and re-derived rather than accepted. `rg DEFAULT_CONFIG_REPO`
    across the worktree outside `tasks/` finds exactly the four sites the
    finding names: the facade's import line and `__all__` entry, the definition
    at `inspector.py:58` and the single use at `inspector.py:95`. `master`'s
    facade did not export it. Dropped from both facade lines; `__all__` is now
    89 names by `len(scufris_host.__all__)`, and DECISION.md's bullet reads
    four added names.
- [x] R4.4 (NIT) packages/host/src/scufris_host/__init__.py:39 - the module
  docstring says keeping `render` a module "keeps fifteen renderers out of the
  flat namespace"; `render.py` defines fourteen `render_*` functions, which is
  what `examples/host_report_fixture.py:6` says and what
  `test_host_report_fixture_calls_every_renderer` asserts over. Change
  "fifteen renderers" to "fourteen renderers".
  - Response: fixed. `rg -c '^def render_' packages/host/src/scufris_host/render.py`
    -> 14, which is what the close-out already recorded for the plan's own
    "seventeen" and what the example's docstring says.

Process signal: commit `b91a8d0`'s message still records the method R2.1
disproved ("Confirmed the fix under `python -P` with `PYTHONPATH` cleared").
History is immutable and there is nothing to change, but it is a fourth
surface carrying a refuted claim while only the two task records were
corrected. A correction pass owes a look at every surface the claim was
written to, not only the ones the finding cited.

Process signal: four rounds, and every finding has been prose rather than
code - three methods and two counts. R4.1 is the fourth count on a branch
whose own retro already names the cure ("paste the command and its exit
code"). The counts in DECISION.md were never re-derived after the facade
changed shape; two `rg -c` runs would have caught it before round 1.

Verified by the recording pass, independently of the out-of-context reviewer:
`ruff check .`, `ruff format --check .`, `mypy scufris packages` (154 files)
and `python -m pytest` (exit 0) in `nix develop`; all five proofs by exit
code on their own criterion; `nix flake check`,
`nix build .#scufris .#scufris-web`, the three check derivations and
`nix build .#scufris-vm-test`, all exit 0. Falsified the boundary proof by
inserting `from scufris_host.run import Runner` into `scufris/checks.py` and
watching `test_no_package_imports_a_sibling_private_module` go RED, then
restored the tree. Re-derived every round-4 finding from the tree rather than
accepting it: the `__all__` counts by `rg -c` on both revisions, the
`render_*` count by `rg -c "^def render_"`, the stale comment and the
`DEFAULT_CONFIG_REPO` reference set by `rg` across the worktree. Also
re-derived R3.1's fix end to end. Doc sweep outside `tasks/` finds no other
stale `scufris.host` reference, and no root module imports `psutil`. No
`manual:` proofs are outstanding.

## Round 5

- REVIEWER: out-of-context
- VERDICT: APPROVE

All four round-4 findings are confirmed fixed and their boxes are ticked.
Re-derived independently of the out-of-context reviewer rather than accepted:
an AST walk over `__all__` gives 63 on `master` and 89 on the branch with
nothing removed, and the `from .metrics import` / `from .processes import`
blocks carry 16 and 6, so 22 re-exports leave the four added names DECISION.md
now records; `rg DEFAULT_CONFIG_REPO` outside `tasks/` finds only
`inspector.py:58` and `:95`; `rg -c '^def render_'` -> 14 against the
docstring's "fourteen"; and `tests/test_host_actions.py:51` no longer names a
module path. The fix commit `9090ed8` touches two facade lines and one comment,
nothing outside the package referenced the dropped name, and the gate is green
after it.

One new finding, and it is the only live surface any of the five rounds' doc
sweeps still had open. It does not block: no BLOCKER or MAJOR is open.

- [ ] R5.1 (NIT) CHANGELOG.md:111 - the `[Unreleased]` documentation-restructure
  entry points readers at `scufris/host/README.md`, a path this branch moved.
  `[Unreleased]` runs from line 8 to line 465, so the dead path ships as v0.2.0
  release notes. Independently confirmed: a slash-form sweep
  (`scufris/host/`, `scufris/metrics`, `scufris/processes`) outside `tasks/`
  returns this line and nothing else, which is why four rounds of sweeping the
  dotted `scufris.host` form never saw it. Change to
  `packages/host/src/scufris_host/README.md`, the path `README.md:31` and
  `AGENTS.md:19` already use. Graded NIT rather than a Step failure: the Step
  enumerates the files it wanted updated and every one of them was, so this is
  sweep hygiene rather than an undelivered clause - but it is a one-token edit
  worth taking before landing.
  - Response:

Process signal: five rounds, and rounds 2 through 5 cost four cycles for two
prose corrections, one unrequested export and one stale path. Every finding
after round 1 was a claim ABOUT the code - a count, a method, a path - rather
than the code itself, and each was found by a different grep form than the one
the previous sweep used. The dotted-versus-slash miss in R5.1 is the same shape
as R4.1's uncounted `__all__`: a sweep is only as complete as the spelling it
searches for, and no round wrote down which spellings it had covered.

Verified by the recording pass, independently of the out-of-context reviewer:
`ruff check .`, `ruff format --check .`, `mypy scufris packages` (154 files) and
`python -m pytest` (1117 passed, 1 skipped) in `nix develop`, all exit 0; all
five proofs run bare and green on their own criterion, including
`uv run python -c "import scufris_host"`, the boundary test, the
`packages/host/tests` + `--collect-only | rg -q packages/host` pair, the offline
example and `test_stats_endpoint_matches_inspector_output`; `tatr check` exit 0.
Re-derived every round-4 fix from the tree. Not verified in this round:
`nix build .#scufris .#scufris-web` and the KVM-only VM tests, both of which
round 4's recording pass ran green on the same code. No `manual:` proofs are
outstanding.
