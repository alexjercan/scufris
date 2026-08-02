# Review: Move the configuration-change registry onto the database

- TASK: 20260803-002141
- BRANCH: refactor/config-change-registry-db

## Round 1

- REVIEWER: out-of-context
- VERDICT: APPROVE

- [ ] R1.1 (MINOR) tests/test_nixos_config_change.py:567 - the DoD's central
  proof does not reopen the database. `make_client` holds the first
  `TestClient` open for the whole test and `create_app` takes its handle from
  the process-wide memo (`app.py:960` -> `db/__init__.py:45`), so the
  "restarted" app is handed the SAME `Database` and pool. It does prove the
  registry is no longer a per-app dict (404 on the base), but not that the row
  is committed and readable through a freshly opened engine, which is what the
  docstring's "outlive the process" claims.
  `test_the_digest_store_survives_a_restart_and_stays_bounded`
  (tests/test_host_digest.py:165) is the repo's pattern for this and says why:
  "Reopened rather than shared". Exit the first client, or call
  `close_state_database(state_dir)`, before building the restarted app; same
  for `test_a_build_interrupted_by_a_restart_does_not_block_the_repo:601`.
  - Response:
- [ ] R1.2 (MINOR) scufris/hostconfig/changes.py:174 - `_reap` is rewritten as
  SQL (`ORDER BY (state == 'building') ASC, seq ASC LIMIT over`) and nothing
  anywhere exercises it; `max_changes` has no caller but the default, so a
  silent no-op would grow the table without bound and no check would notice.
  Add a store-level test with `max_changes=3` asserting a settled change drops
  before a building one, and that the oldest goes when all are building.
  Behaviour is unchanged from master's in-memory `_reap`, and the untested
  bound is pre-existing, which is why this is not a MAJOR; it is now durable
  state, which is why it is worth closing here.
  - Response:
- [ ] R1.3 (NIT) scufris/hostconfig/changes.py:146 - `abandon_builds()` returns
  a count that its only caller (`app.py:1755`) discards and no test asserts.
  Either drop the `int` return per YAGNI, or log it at startup so the sweep is
  observable.
  - Response:

Verified independently of the round-1 reviewer: `ruff check .` clean, `mypy .`
clean (192 files), `python -m pytest` 973 passed, exit 0, all from the
worktree. R1.1's mechanism re-derived from `app.py:960`, `db/__init__.py:45-70`
and `conftest.py:327` rather than taken on report. R1.2's "behaviour unchanged"
re-derived from master's `changes.py:94-101`. Migration `e054a39a5fae` chains
off the prior head with symmetric upgrade/downgrade, and the autogenerate-diff
test is green. Every DoD `cmd:` proof holds.

Not verified: a real multi-process restart or a crash mid-build against a live
NixOS flake - the example cannot get past the attribute probe without a real
flake, on the branch and on the base alike.

- Process signal: the round-1 reviewer found, and I reproduced on master, that
  `create_app` is unrunnable from a running event loop since 18b117b -
  `python examples/comms_loop.py` exits 1 with "a transaction cannot be opened
  on a thread with a running event loop" from `sessions.prune` (`app.py:1107`).
  `examples/telegram_approval.py:125` and `examples/telegram_bot.py:134` have
  the same shape. Pre-existing and outside this diff; filed as its own task
  rather than held against this branch.
- Process signal: `_row` / `_values` / `_change` / `_reap` are now a second
  near-verbatim copy of `host_actions.py:363-403`. The plan chose that
  deliberately, and at two instances it is still the right call, but a third
  store would argue for a shared row-store helper.
