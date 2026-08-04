# Fix the examples the package carve broke

- PRIORITY: 103
- TAGS: bug, v0.2.0, examples
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As the Scufris maintainer, I want the carve's stale test path out of
`examples/host_agent.py` and `examples/telegram_approval.py`, and
`host_agent.py` back on the gate, so that the runnable examples the README
points at are not silently broken by the package carve.

`6d998c8` (the hostd carve) moved `tests/test_host_actions.py` to
`packages/hostd/tests/test_host_actions.py`. Both examples do their own
`sys.path` setup - `sys.path.insert(0, str(ROOT / "tests"))`
(`examples/host_agent.py:43`, `examples/telegram_approval.py:39`) - and import
`host_files` / `host_runner` from that module, so both now die at import with
`ModuleNotFoundError: No module named 'test_host_actions'`.

The pytest suites that import the same module still work, because pytest's own
collection puts `packages/hostd/tests` on the path. Only the examples broke,
and neither was on the `OFFLINE` opt-in tuple that `tests/test_examples.py`
gates, so nothing caught it.

Found while planning 20260803-214750, which needs `examples/host_agent.py` to
run as a DoD proof.

Planning measured the two examples rather than assuming one defect, and found
two. The path fix alone takes `host_agent.py` green. `telegram_approval.py`
then gets past import and dies at `create_app` (line 127) on the event-loop
transaction guard - a separate defect already filed as `20260803-014210`, which
names this file as one of its three call sites. This task fixes the carve
regression in both files and gates `host_agent.py`; `telegram_approval.py`
goes green and joins the gate under `20260803-014210`. See DECISION.md.

## Steps

- [x] Point `examples/host_agent.py:43` and `examples/telegram_approval.py:39`
      at `ROOT / "packages" / "hostd" / "tests"` instead of `ROOT / "tests"`.
      Keep the plain path insert: `host_files` and `host_runner` are module
      level helper functions, not pytest fixtures, and nothing in
      `packages/hostd/src` exports them - one consumer pair does not earn a
      package export.
- [x] Add `"host_agent.py"` to the `OFFLINE` tuple in
      `tests/test_examples.py:32`, keeping the tuple's ordering convention.
      Do NOT add `telegram_approval.py` - it is still red on the event-loop
      guard, and `20260803-014210` adds it when it lands.
- [x] Confirm no other example carries a stale carve path. Planning already
      ran all thirteen under a freshly synced env: nine green, `host_agent.py`
      and `telegram_approval.py` on the stale path, and `comms_loop.py` plus
      `telegram_bot.py` on the event-loop guard that `20260803-014210` owns.
      Re-run the sweep and reconcile against that expectation rather than
      chasing anything new.
- [x] Append a line to `20260803-014210`'s Notes recording that
      `telegram_approval.py`'s path is now correct, so its fix is the only
      thing left between that file and the `OFFLINE` tuple.

## Definition of Done

- `examples/host_agent.py` runs green
  (cmd: `uv run python examples/host_agent.py`).
- The offline gate runs `host_agent.py`, so the next carve that breaks it fails
  the suite (cmd: `uv run pytest tests/test_examples.py -k host_agent`).
- `examples/telegram_approval.py` reaches `create_app` instead of dying at
  import, proving the carve regression is gone from it too
  (cmd: `uv run python examples/telegram_approval.py 2>&1 | grep -q
  "cannot be opened on a thread with a running event loop"`).
- The example sweep matches the expectation above, with the only failures being
  the three the event-loop task owns (manual: run
  `for f in examples/*.py; do uv run python "$f" >/dev/null 2>&1 ||
  echo "FAIL $f"; done` and confirm it prints exactly `comms_loop.py`,
  `telegram_bot.py` and `telegram_approval.py`).
- The whole suite stays green (cmd: `uv run pytest -q`).

## Notes

- Verify in a FRESHLY SYNCED environment; run `uv sync` first. A stale `.venv`
  missing `_editable_impl_scufris_hostctl.pth` makes five more examples fail
  with `ModuleNotFoundError: No module named 'scufris_hostctl'`, and a bare
  `python` outside the env fails eight. Neither is a repo defect -
  `pyproject.toml:53` and `uv.lock` both carry the package. Every proof above
  is written `uv run` for this reason.
- The examples gate already exists and already runs under `nix flake check`:
  `tests/test_examples.py` runs each `OFFLINE` entry as a subprocess, and
  `flake.nix:261` runs `python -m pytest`. The record's original third Step
  asked whether to build one; the answer is that it is built and
  `host_agent.py` was simply never enrolled.
- Proving the tuple covers EVERY workspace member is `20260804-053002`'s Done
  Means 3 (`test_every_package_has_a_gated_example`), not this task.
- `host_agent.py` runs offline in ~1.7s, well inside the gate's 120s timeout.
- Both `cmd:` proofs confirmed red on the base: the example exits 1 with
  `ModuleNotFoundError`, and the `-k host_agent` selection exits 5 with
  `5 deselected` because nothing collects.

## Close-out

**What and why.** Three one-line edits, exactly as planned. Both examples'
`sys.path.insert` now points at `ROOT / "packages" / "hostd" / "tests"`, the
directory `6d998c8` moved `test_host_actions.py` to. `"host_agent.py"` joins
the `OFFLINE` tuple in alphabetical position, between `core_unit_of_work.py`
and `host_report_fixture.py`, matching the tuple's existing ordering. A Notes
line on `20260803-014210` records that `telegram_approval.py`'s path is now
correct, so its event-loop fix is the only thing left before that file can be
enrolled too.

**Alternatives.** None reopened. DECISION.md settled scope before
implementation: no package export of `host_files` / `host_runner`, no second
gate in `flake.nix`, and no folding in the event-loop fix. The code confirmed
the premises - `host_files` and `host_runner` are module-level helpers, and
`tests/test_examples.py` already runs each `OFFLINE` entry as a subprocess
under a 120s timeout.

**Difficulties.** None. The reds reproduced exactly as the plan predicted, and
the sweep matched the predicted failure set with no surprises.

**Evidence.** Red on base: `examples/host_agent.py` exit 1 with
`ModuleNotFoundError: No module named 'test_host_actions'`; the `-k host_agent`
selection exit 5, nothing collected. Green after: the example exits 0 and
prints the deny path; `pytest tests/test_examples.py -k host_agent` collects
and passes one test; `telegram_approval.py` reaches `create_app` and the
event-loop grep matches. The sweep over all thirteen examples prints exactly
`comms_loop.py`, `telegram_approval.py`, `telegram_bot.py`, the three
`20260803-014210` owns. `uv run pytest -q` exits 0. `grep -rn 'ROOT / "tests"'
examples/` finds nothing.

**Reflection.** The gate existed and the entry did not, which is the failure
mode a hand-written opt-in tuple invites. `20260804-053002`'s
`test_every_package_has_a_gated_example` is the durable fix; until it lands,
every new example depends on someone remembering this tuple.
