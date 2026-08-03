# Review: Move the root helper into packages/hostd

- TASK: 20260803-214747
- BRANCH: refactor/hostd-package

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

The carve itself is clean and the guards bite. Both load-bearing claims were
re-derived by mutation in the recording pass: pointing `scufris/checks.py` at
`scufris_hostd.actions` fails `test_no_package_imports_a_sibling_private_module`,
and bumping the root pin to `scufris-hostd==0.2.0` fails
`test_the_app_pins_hostd_to_one_exact_version`. `uv.lock` records only
`{ name = "scufris-hostd", editable = "packages/hostd" }`, confirming
DECISION.md's claim that uv drops the specifier. Every finding below is a record
or documentation defect, not a code one.

- [x] R1.1 (MAJOR) CHANGELOG.md:19 - "a deployment pinning `package` by hand
  needs no change and one built from the flake picks it up" states the breaking
  case backwards. `packages.scufris` no longer carries `bin/scufris-hostd`
  (verified: `nix build .#scufris` then `ls result/bin` is `scufris` alone) and
  `nix/scufris-hostd.nix:146` execs `${cfg.package}/bin/scufris-hostd`, so an
  operator who pinned `package = scufris.packages.<system>.scufris` gets a root
  unit whose `ExecStart` does not exist. The default is the case that needs no
  change. Replace the clause with: a deployment that pinned `package` to
  `packages.scufris` MUST re-point it at `packages.scufris-hostd`; one on the
  default picks it up.
  - Response: fixed. CHANGELOG.md:19-23 now says a deployment on the default picks it up, while one that pinned `package` to `packages.scufris` MUST re-point it at `packages.scufris-hostd`, and names the missing `bin/scufris-hostd` as the reason.
- [x] R1.2 (MINOR) packages/hostd/src/scufris_hostd/README.md:40 - the options
  table still documents `package`'s default as
  `scufris.packages.<system>.scufris`, which `nix/scufris-hostd.nix:47-48`
  changed in this diff. Change the cell to
  `scufris.packages.<system>.scufris-hostd`.
  - Response: fixed. The options table cell is now `scufris.packages.<system>.scufris-hostd`, matching `nix/scufris-hostd.nix:47-48`.
- [x] R1.3 (MINOR) packages/hostd/src/scufris_hostd/README.md:13 - the move
  broke three relative links that resolved correctly from `scufris/hostd/`:
  `../README.md` (:13) now points at `packages/hostd/src/README.md`, and
  `../../tasks/20260729-125020/DECISION.md` (:15) and
  `../../tasks/20260729-125035/DECISION.md` (:17) at `packages/hostd/tasks/...`.
  Confirmed against `master`, where all three resolve. Re-point them at
  `../../../../scufris/README.md` and `../../../../tasks/<id>/DECISION.md`.
  `../host/README.md` at :201 is broken too, but it was already broken by
  09cf946, so it is not this diff's - fix it in the same pass or leave it.
  - Response: fixed. The three links are re-pointed at `../../../../scufris/README.md` and `../../../../tasks/<id>/DECISION.md`. Fixed `../host/README.md` at :201 in the same pass too, as `../../../host/src/scufris_host/README.md`. All four re-checked with `test -e` from the README's directory.
- [x] R1.4 (MINOR) tasks/20260803-214747/TASK.md:148 - the DoD's regression
  guard names `nix build .#checks.x86_64-linux.scufris-hostd-vm-test`, and that
  attribute does not exist: `nix eval .#checks.x86_64-linux --apply
  builtins.attrNames` is `[ "filesize" "mypy" "pytest" "records" "ruff" ]`, and
  the VM tests live in `packages`. `tatr proofs` therefore emits an unrunnable
  command. The close-out discloses this at :215 rather than hiding it, so it is
  not an honesty finding - but the DoD line is the one a later reader runs.
  Amend it to `nix build .#scufris-hostd-vm-test`, which passes.
  - Response: fixed. The DoD regression guard now reads `cmd: nix build .#scufris-hostd-vm-test`; re-run and green. The close-out note now records the correction rather than the standing discrepancy.
- [x] R1.5 (NIT) packages/hostd/src/scufris_hostd/README.md:371 - section 8
  cites only `examples/host_action.py`. Add a sentence naming
  `examples/hostd_socket_roundtrip.py` as the socket-boundary proof; it is the
  evidence this task added for exactly the boundary this README documents.
  - Response: fixed. Section 8 now names `examples/hostd_socket_roundtrip.py` as the socket-boundary proof and says what it drives.
- [x] R1.6 (NIT) tasks/20260803-214747/TASK.md:229 - "all 8 checks pass (ruff,
  mypy, pytest, records, filesize, build)" names six and counts eight;
  `nix flake check` reports "running 7 flake checks" over the five `checks`
  attributes plus the package builds. Change it to match what the run prints.
  - Response: fixed. The evidence line now reads exit 0 / "all checks passed!", and states the cold-store "running 7 flake checks" as the five `checks.x86_64-linux` attributes plus the two package builds.
- [x] R1.7 (NIT) tests/test_domain_routers.py:39 - imports the underscore-private
  `_record` from `tests/domain_router_fakes.py:71` across a module boundary. Now
  that the helper is a module rather than file-local scaffolding, rename it to
  `record` at its definition and at `test_domain_routers.py:487`.
  - Response: fixed. `_record` renamed to `record` at `tests/domain_router_fakes.py:71` and at both use sites in `test_domain_routers.py` (:39 import, :487).

Plain observations, not findings:

- Every DoD proof was run in the recording pass and passes on its stated
  criterion: `import scufris_hostd`; `test_the_app_pins_hostd_to_one_exact_version`;
  `pytest packages/hostd/tests` plus the collect grep (3 hits);
  `test_no_package_imports_a_sibling_private_module`; `nix build .#scufris-hostd`
  with `bin/scufris-hostd` present and `.#scufris` carrying `bin/scufris` alone;
  `pytest tests/test_examples.py -k hostd`; `nix build .#scufris-hostd-vm-test`;
  `nix flake check` exit 0. Full suite exit 0 and `tatr check` clean.
- No `manual:` proofs are open, so there are no pending user checks.
- `_import_roots()` in `tests/test_package_boundaries.py:173` does include
  `scufris/`, so the DoD's facade claim is genuinely enforced against root
  modules and not only between members.
- A markdown link sweep over `README.md`, `scufris/README.md`, `AGENTS.md`,
  `docs/RELEASING.md` and `packages/host/src/scufris_host/README.md` found no
  broken target; the moved helper README (R1.3) is the only one.
- `cap_for` gives `packages/hostd/tests/*.py` the test cap via its `/tests/`
  branch, so the 880-line moved module is correctly capped.
- Not verified: the published-wheel `Requires-Dist` claim (no wheel was built)
  and non-Linux flake outputs.
- Process signal: decisions 5a and 5b restructured root tests beyond the plan -
  `tests/test_domain_routers.py` shed 329 lines into `tests/domain_router_fakes.py`
  because an import rewrite pushed it one line over the file cap. Both are
  recorded in DECISION.md and the move is verbatim, so this is evidence about
  planning near a ratchet, not a YAGNI finding.
- Process signal: `examples/hostd_socket_roundtrip.py:23` says the script
  exercises "the refusal to run anything that was not previewed and approved";
  it drives only the happy path. Wording, and no assertion is weakened by it.

Inspection commands:

```sh
cd "$(sprout show refactor/hostd-package)"
git diff master...HEAD
uv run --no-sync python -m pytest -q
nix flake check && nix build .#scufris-hostd-vm-test
nix build .#scufris .#scufris-hostd && ls result/bin
```

## Round 2

- REVIEWER: out-of-context
- VERDICT: APPROVE

All seven round-1 findings verified fixed against the tree, not against their
Response lines. The recording pass re-derived the R1.3 link claim itself
(`test -e` on all four targets from the README's own directory: all resolve)
and re-ran both gates. `uv run --no-sync python -m pytest -q -o addopts=""` is
exit 0, 1119 passed 1 skipped; `nix flake check` is exit 0, "all checks
passed!". The fix commit touches no production code beyond the `_record`
rename, and the four findings below are all cosmetic residue of the fix pass
itself.

- [ ] R2.1 (NIT) tasks/20260803-214747/TASK.md:232 - the R1.6 fix replaced one
  wrong gloss with another. "the two package builds" is not what the seven
  realised derivations are: a cold-ish run builds `scufris-0.1.0`,
  `scufris-dev-env` and the five `checks.x86_64-linux` attributes, and
  `.#scufris-hostd` / `.#scufris-web` are not among them
  (`nix eval .#packages.x86_64-linux --apply builtins.attrNames` lists seven
  package attributes, so "the two package builds" has no referent either).
  Drop the gloss and keep only what the run prints, or change it to "plus the
  app package and the dev shell".
  - Response: fixed. The gloss is dropped; the evidence line now records only exit 0 and "all checks passed!", which is what the run prints.
- [ ] R2.2 (NIT) tasks/20260803-214747/DECISION.md:63 - decision 5b still names
  the moved helper as `` `_record`/`_change` ``. R1.7 renamed it, so the record
  now describes a symbol that does not exist. Change it to `record`/`_change`.
  - Response: fixed. DECISION.md:63 now reads `record`/`_change`.
- [ ] R2.3 (NIT) CHANGELOG.md:22 - the R1.1 rewrite left a 98-column line in a
  file otherwise wrapped at ~80. Re-wrap the paragraph.
  - Response: fixed. The paragraph is re-wrapped; no line in CHANGELOG.md exceeds 82 columns.
- [ ] R2.4 (NIT) packages/hostd/src/scufris_hostd/README.md:202 - the longer
  link target pushed this prose line to 105 columns; the other >90 lines in
  this file are all tables. Re-wrap.
  - Response: fixed. The sentence is split across two lines; :202 is now 63 columns.

Plain observations, not findings:

- The R1.4 fix is verified beyond the text: `tatr proofs 20260803-214747` now
  emits `nix build .#scufris-hostd-vm-test`, and that attribute is in
  `packages.x86_64-linux`. The build was re-run green in the fix pass.
- The `_record` -> `record` rename is complete: no `_record` remains under
  `tests/` or `packages/`. The other `_record` hits in `scufris/` are unrelated
  private helpers in different modules.
- R1.3 got one more fix than it asked for: the pre-existing `../host/README.md`
  break from 09cf946 was repaired in the same pass, as the finding permitted.
- No `manual:` proofs are open, so there are no pending user checks.
- The four NITs are open, not ticked. None is a BLOCKER or MAJOR, so none
  blocks this APPROVE; they are cheap enough to sweep in the compound pass or
  to leave.
