# Review: Split the oversized test suites under the size cap

- TASK: 20260731-171432
- BRANCH: refactor/split-test-suites

## Round 1

- REVIEWER: out-of-context
- VERDICT: APPROVE

- [ ] R1.1 (MINOR) scufris/app.py:1320 - the `docs:` commit `3fe5be1` carried a
  whole-file `ruff format` pass (~13 unrelated re-wrap hunks at 1320, 1507,
  1542, 1607, 1629, 1643, 1663, 1678, 2441, 2605, 2671, 2680, 2757) into a file
  the task Notes name as out of scope and 20260729-103712 owns, so that task
  now rebases across a reformat it did not ask for. Restrict the commit to the
  docstring repoint at 1091 and let the owning task take the format pass.
  - Response:
- [ ] R1.2 (MINOR) tests/conftest.py:181 - the `_Helper` section comment names
  only `test_host_action_api.py` and `test_telegram_approvals.py` as consumers,
  but the fixture is now imported by six modules (`test_host_action_decisions.py`,
  `test_nixos_config_change.py` and `test_host_digest.py` too), so the comment
  understates the blast radius of a change to it. Replace the enumeration with
  "every module that drives the privileged path for real".
  - Response:
- [ ] R1.3 (NIT) tests/test_host_mcp_server.py:132 - the comment here and the
  same sentence at line 11 say the inspection parsers are "pinned against
  captured fixtures in `test_host_inspection.py`", but C4 moved the thermal and
  nix-store parser pins into `test_host_thermal.py` and
  `test_host_nix_store.py`. Add both file names at each site, the way commit
  `3fe5be1` repointed the auth citations.
  - Response:

Process signal: the per-commit move-proof plus name-set-difference rig is why
this round found no correctness defect - every recorded number reproduced on an
independent rig. Worth promoting to a standing convention for move refactors.

Process signal: the Step "`ruff format <the files you edited>` - scoped, never a
whole dir" has a gap. A one-line edit to an un-formatted legacy file still drags
in a whole-file reformat, because `ruff format` has no hunk scope. The rule owes
a second clause: revert format-only hunks in files outside the task's scope.

Verified in-session, independently of the round-1 reviewer:

- `python -m pytest --collect-only` nodeids with the `<file>::` prefix stripped,
  collected separately on `master` and on the branch: 896 on each, `diff` empty.
  The load-bearing re-derivation for this round.
- `python scripts/check_file_size.py` exit 0; `ALLOWLIST` is exactly
  `{scufris/app.py, tests/test_app.py}` and `check()` returns `[]`.
- `git rebase master --exec 'python scripts/check_file_size.py'` green across
  all 10 commits, tip hash `47c1cbb` unchanged.
- `nix flake check` evaluated all five checks (`pytest`, `records`, `filesize`,
  and the two package builds) with every derivation already in the store;
  `nix build .#scufris .#scufris-web` green; `cd web && npm run ci` green.
- The DoD task-ID grep over the split files prints nothing.
- The five non-test files in the diff: `scripts/check_file_size.py` deletes
  exactly the eight allowlist entries the Steps name; `scufris/auth/policy.py`,
  `scufris/README.md` and `examples/auth_session.py` are citation repoints;
  `scufris/app.py` is one citation repoint plus the reformat in R1.1. A
  `git diff -w` over `app.py` leaves only re-wraps - no behavior change, and the
  file is allowlisted so the guard is unaffected (3764 -> 3745 lines).

Not verified: the claimed per-commit re-measurement of the four near-cap
siblings - only the tip is measurable after the fact, and it is green there.

No open `manual:` proof on this task.
