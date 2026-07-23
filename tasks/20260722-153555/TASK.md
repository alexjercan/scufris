# Green the mypy gate: enable pydantic.mypy plugin + fix enum-typed test args

- STATUS: CLOSED
- PRIORITY: 15
- TAGS: chore, tests, mypy

## Story

The repo's mypy gate (`nix flake check` -> `mypy .`) is RED on master with 44
pre-existing `[arg-type]` errors in the test suite, discovered during
20260722-135525. They are NOT a regression from any recent feature - master
fails identically. Two distinct causes:

1. `Settings(agent_backend="app_server")` etc. - a `str` passed to a pydantic
   field typed as a `StrEnum` (`Backend`/`AuthMode`). pydantic coerces these at
   runtime (the field validators exist), but mypy does not know that WITHOUT the
   pydantic mypy plugin. ~38 of the 44.
2. `store.mark_finished(..., state="done")` - a `str` passed to a plain (non
   -pydantic) method whose param is typed `AgentState`. The plugin does NOT fix
   these; the calls should pass the enum member. ~6 of the 44, in
   test_agent_store.py, test_mcp_server.py, test_app.py.

## Steps

- [ ] Add `plugins = ["pydantic.mypy"]` under `[tool.mypy]` in pyproject.toml.
- [ ] Re-run `nix develop --command mypy .`; confirm the ~38 `Settings(...)`
      arg-type errors clear.
- [ ] Fix the remaining ~6 `mark_finished(state="...")` calls to pass the
      `AgentState` enum member (import it in the test module).
- [ ] Confirm `mypy .` and `nix flake check` are fully green.

## Definition of Done

- `nix flake check` passes the mypy check (cmd: `nix build .#checks.x86_64-linux.mypy`).
- No new `# type: ignore` added to source to achieve it (the plugin + enum use
  is the proper fix), verified by review.

## Notes

- Discovered in 20260722-135525 (opencode backend). That branch is at exact
  parity with master (44 == 44), so it did not introduce these.
- The pydantic plugin experiment during 135525 confirmed it drops 44 -> 6; the
  6 remaining are the mark_finished calls above.

## Resolution (SUPERSEDED)

Closed 2026-07-23 as SUPERSEDED by 20260723-182253 (commit 14ec355, "type-correct
the enum-vs-str call sites so mypy is green again"), which greened the gate the
next day by a more complete and deliberately-chosen method: it fixed ALL the
enum-vs-str call sites to pass the typed enum member (`Backend`/`AuthMode`/
`AgentState`) - including the `mark_finished(state=...)` calls this task listed -
rather than enabling the `pydantic.mypy` plugin. That approach is now the durable
lesson `strenum-fields-take-the-member-not-the-raw-str-in-typed-callers`; the old
`scufris-mypy-baseline-is-red` lesson is marked RESOLVED.

Both DoD criteria are already satisfied on master with no further work:
- `nix build .#checks.x86_64-linux.mypy` -> EXIT 0 (verified 2026-07-23).
- `mypy .` -> "Success: no issues found in 47 source files".
- No `# type: ignore` added; no `plugins` line needed in pyproject.toml.

Enabling the `pydantic.mypy` plugin was deliberately NOT done: the tree is already
green, so the plugin would only change pydantic type-check semantics repo-wide
against the prior decision, for no current benefit. If ever wanted, it should be
its own scoped task justified on catching a real class of bug.
