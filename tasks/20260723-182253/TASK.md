# Fix pre-existing mypy baseline: Literal-vs-str in test files

- PRIORITY: 22
- TAGS: chore, typing, tests
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As a developer trusting the QA gate, I want `mypy .` (and `nix flake check`'s mypy
check) to be green again, so a task DoD that says "mypy green" can be taken
literally instead of "adds no NEW errors". Today the gate is red on master with a
large pre-existing baseline that hides any new type error under the noise.

## Context (grounded)

`nix flake check`'s mypy check (`mkCheck "mypy" "mypy ."` in `flake.nix:186`) is RED
on master (ce8e441): 58 errors across 6 test files. Nearly all are the same shape -
a test passes a plain `str` where a `Literal` is expected:

- `Settings(agent_backend="codex")` where `agent_backend: Backend`
  (`tests/test_app.py`, many lines);
- `Settings(agent_claude_auth_mode="...")` where `AuthMode` is expected
  (`tests/test_app.py:1572`);
- `AgentStore.mark_finished(state="...")` where `AgentState` is expected
  (`tests/test_app.py:2382`, `tests/test_agent_store.py` many lines, and
  `tests/test_agent_store.py:696` a `RunOutcome | None` union-attr).

Discovered during SC1 (`tasks/20260723-153609`): the SC1 change is mypy-clean, but
the baseline made "mypy green" impossible to satisfy literally. See that task's
RETRO.md and lesson `scufris-mypy-baseline-is-red`.

## Steps

- [x] Enumerate the 58 errors: `nix develop -c bash -c 'mypy .'` and group by shape.
- [x] Fix each at the call site by using the typed value rather than a bare string:
      cast to the Literal / import and use the enum-like type (`Backend`, `AuthMode`,
      `AgentState`), or annotate the local. Prefer the fix that makes the test state
      the real type, not a blanket `# type: ignore`.
- [x] Handle the `RunOutcome | None` union-attr at `test_agent_store.py:696` with an
      explicit assert-not-None (or a typed helper) so the attribute access is safe.
- [x] Re-run `nix develop -c bash -c 'mypy .'` to 0 errors.

## Definition of Done

- `mypy .` reports 0 errors on the whole tree. (cmd: `nix develop -c bash -c 'mypy .'`)
- `nix build .#checks.x86_64-linux.mypy` succeeds. (cmd)
- `ruff check .` and `python -m pytest` still green. (cmd: `python -m pytest`)

## Notes

- Pure test-code typing cleanup; no runtime behavior change.
- Lesson: `scufris-mypy-baseline-is-red` (SC1 retro).
