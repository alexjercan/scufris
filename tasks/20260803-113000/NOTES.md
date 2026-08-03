# Notes: Prove the startup sweep clears a building row orphaned by a crash

## What changes

Nothing user-visible. `abandon_builds` is already correct; what is missing is a
proof that it clears the state it actually exists for.

- Before: `tests/test_nixos_config_change.py` proves the sweep clears a
  `building` row that ANOTHER LIVE PROCESS is still building
  (`test_a_build_interrupted_by_a_restart_does_not_block_the_repo`, line 619:
  its first `TestClient` is never exited, so the hanging build still hangs).
  No test covers a row left by a process that died without running its
  shutdown hooks - SIGKILL, OOM, power loss.
- After: one added test walks the crash case end to end - a first process runs
  and exits cleanly, a `building` row is then re-established through
  `ConfigChangeStore` against the same state directory (which is what a SIGKILL
  leaves behind), that store's handle is closed, and the restarted app's
  startup sweep fails the row, with a reason, and the repository takes a new
  build instead of a 409 nothing can clear.

The existing test stays byte-identical. It covers the live-process case and is
an addition's sibling, not its predecessor.

## Surfaces

| File | Why |
|-|-|
| `tests/test_nixos_config_change.py` | the only file changed: one new test after `test_a_build_interrupted_by_a_restart_does_not_block_the_repo`, plus two imports |
| `scufris/hostconfig/changes.py` | read only: `abandon_builds` (146), `building_for` (127), `ConfigChangeStore.put` (88) |
| `scufris/app.py` | read only: `abandon_builds()` at startup (423), `close_state_database` in the lifespan (242) |
| `scufris/db/__init__.py` | read only: `state_database` memoizes one handle per resolved state dir (45); `close_state_database` closes AND evicts (73) |
| `tasks/20260803-014401/DECISION.md` | the record this task is seeded by; the new test is its second alternative |

No production change is expected. If one turns out to be needed, that is a stop.

## Data and interfaces

Nothing added or changed. What the test consumes, all already exported:

- `ConfigChangeStore(db: Database, *, max_changes: int = MAX_CHANGES) -> None`
- `ConfigChangeStore.put(change: ConfigChange) -> ConfigChange`
- `ConfigChange(id: str, resolved: Resolved, attr: str, state: ChangeState = BUILDING, ...)`
- `Resolved(repo: str, ref: str, rev: str, ...)`; `ConfigChange.repo` is `resolved.repo`
- `state_database(state_dir: Path) -> Database` / `close_state_database(state_dir: Path) -> None`
  (`scufris/db`, both currently unimported by this test module)

`ChangeState`, `ConfigChange`, `ConfigChangeStore`, `Resolved`, `Database` and
`TestClient` are already imported at the top of the module.

The `repo` string must equal what `resolve` writes (`scufris/hostconfig/resolve.py:144`,
`repo=str(main)` - the resolved main worktree path), or `building_for` will not
match and the 409-cleared assertion proves nothing. Take it from a real change
the first process created over HTTP (`before["resolved"]`) rather than
hardcoding `str(config_repo)`; a hardcoded path that stops matching turns the
test green for the wrong reason.

## Sketches

Illustrative only.

```python
def test_a_build_orphaned_by_a_crash_is_swept_at_the_next_startup(...):
    """..."""
    first = _app(tmp_path, fake_collector, helper, config_repo)
    with TestClient(first) as client:
        csrf = _login(client)
        resp = _post(client, csrf, "/api/host/config/changes", ref="config/add-ripgrep")
        built = _settle(client, csrf, resp.json()["id"], want="proposed")

    # A crash, not a shutdown. The clean path CANNOT produce this row: the build
    # generator's cancellation handler writes `cancelled` before re-raising
    # (scufris/hostconfig/changes.py:329), so `Supervisor.aclose()` leaves
    # `cancelled`, which the sweep neither touches nor needs to. A process
    # killed with SIGKILL runs no handler and leaves the row exactly like this.
    # Do not "simplify" this back to an HTTP build - see 20260803-014401
    # DECISION.md 1.
    orphaned = ConfigChange(
        id="orphaned-by-a-crash",
        resolved=Resolved(**built["resolved"]),
        attr="nixos",
        state=ChangeState.BUILDING,
    )
    ConfigChangeStore(state_database(tmp_path)).put(orphaned)
    close_state_database(tmp_path)   # the crashed process is gone with it

    restarted = make_client(_app(tmp_path, fake_collector, helper, config_repo))
    csrf = _login(restarted)

    swept = restarted.get(f"/api/host/config/changes/{orphaned.id}").json()
    assert swept["state"] == "failed", swept
    assert "restart" in swept["error"]
    assert swept["action_id"] == ""
    # ... and the built change from before the crash is untouched
    assert restarted.get(f".../{built['id']}").json()["state"] == "proposed"
    # ... and the repository is buildable again rather than 409-locked
    again = _post(restarted, csrf, "/api/host/config/changes", ref="master")
    assert again.status_code == 201, again.text
```

## Shape

```
  process 1                       test                      process 2
  ---------                       ----                      ---------
  TestClient(first)
    lifespan up
    POST /changes ---> built row (proposed)
    lifespan down
      close_state_database(tmp_path)   [memo evicted]
                                  state_database(tmp_path)
                                    [fresh handle, re-memoized]
                                  store.put(BUILDING row)
                                  close_state_database(tmp_path)
                                    [memo evicted again]
                                                            _app(...)
                                                              state_database ->
                                                                fresh handle
                                                              abandon_builds()
                                                                BUILDING -> FAILED
                                                            GET  -> failed + reason
                                                            POST -> 201, not 409
```

The whole point sits in the memo: `state_database` hands one handle per resolved
state dir per process, so every hop above must close before the next opens, or
the test proves in-process object sharing instead of a row read back off disk.
This is the same trap `test_a_configuration_change_survives_a_restart` hit in
commit a978f1c.

## Consequences and open questions

- Cost: one more app construction and one more real (faked-executor) build per
  run - the same order as the two restart tests already there. No new fixture.
- What it buys: `abandon_builds` gets a proof of the ONLY state it can ever see
  in production. The current green test's docstring overclaims; this closes the
  gap DECISION.md 1 wrote down rather than deleting the claim.
- What it forecloses: nothing. The live-process test stays.
- The orphan row is written by the TEST, not by a killed process. That is the
  honest limit of an in-process test - `TestClient` cannot be SIGKILLed and
  still leave a usable tmp_path assertion - and the comment must say so, or a
  later reader will think the test forges a state the system cannot reach.
- Open: whether to assert the pre-crash `proposed` row is left alone by the
  sweep. It is a one-line assertion and it discriminates the `WHERE state =
  building` clause from a blanket update, so the sketch includes it; planning
  can drop it if it reads as scope creep.
- Open: whether the new test belongs beside the live-process one in the app
  layer or under the module's third "store" layer. The sketch keeps it in the
  app layer, because the assertion is about what the RESTARTED APP does at
  startup, and only the app layer has that.
