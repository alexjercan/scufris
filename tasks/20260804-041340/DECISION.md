# Decision: fix only the carve regression, and leave the event-loop bug to its own task

- DATE: 20260804-041340
- STATUS: ACCEPTED
- TASK: 20260804-041340
- TAGS: v0.2.0, examples, packaging, scope

## Context

The record was written believing one defect broke both named examples: the
hostd carve moved `test_host_actions.py`, so both examples die at import. That
is true, and it is not the whole story. Running the two under a freshly synced
env, with the path corrected out of band, separates them:

```
$ PYTHONPATH=packages/hostd/tests uv run python examples/host_agent.py
(exit 0)
$ PYTHONPATH=packages/hostd/tests uv run python examples/telegram_approval.py
RuntimeError: a transaction cannot be opened on a thread with a running event loop
```

`telegram_approval.py` gets past import and dies at `create_app`
(`examples/telegram_approval.py:127`). That second defect is already filed as
`20260803-014210`, whose Notes name `examples/telegram_approval.py:125` as one
of its three call sites alongside `comms_loop.py` and `telegram_bot.py`. Its
fix is a design choice - build the app before entering the loop, or offload the
synchronous startup block - that spans all three files.

A full sweep of all thirteen examples under `uv run` confirms the split is
exactly two defects and nothing else: nine green, two on the stale path, two
more on the event-loop guard.

Separately, the record's third Step asks whether to add an examples smoke check
to `flake.nix`. That check already landed. `tests/test_examples.py` runs each
opt-in example as a subprocess, and `flake.nix:261` runs `python -m pytest`
under `nix flake check`. What let `host_agent.py` rot is not a missing gate but
a missing entry: `OFFLINE` (`tests/test_examples.py:32`) is a hand-written
tuple and `host_agent.py` was never on it.

## Decision

This task fixes the carve regression only - the stale `sys.path` line in both
files - and enrolls `host_agent.py` in the existing `OFFLINE` gate.
`telegram_approval.py` keeps its corrected path but stays red until
`20260803-014210` lands, which then enrolls it.

No second gate is built. `host_agent.py` runs offline in ~1.7s, inside the
existing check's 120s timeout.

The fixtures are reached by path insert, not by a package export.
`host_files` and `host_runner` are module-level helper functions in
`packages/hostd/tests/test_host_actions.py`, nothing in `packages/hostd/src`
exports them, and one consumer pair does not earn a public surface.

## Alternatives considered

- **Fold the event-loop fix in, so both named examples go green here.**
  Rejected: it duplicates a filed task, drags in two more examples this task
  never scoped, and turns a two-line path correction into an app-startup
  redesign. The record's original DoD ("both named examples run green") assumed
  one defect; honoring it literally would import the other task's whole design
  choice.
- **Export `host_files` / `host_runner` from `scufris_hostd`.** Rejected as
  speculative: it promotes two test helpers to public API for two callers, and
  the carve deliberately left them in the package's tests.
- **Build a dedicated examples check in `flake.nix`.** Rejected: it would be a
  second gate disagreeing with the one that exists. The `records` check carries
  a comment making exactly this argument - the gate is one thing, not two that
  can disagree.

## Consequences

- The task's DoD changes meaning: `telegram_approval.py` is no longer required
  green here, and is instead proven to fail *past* import, which is what
  distinguishes the carve regression being fixed from it being untouched.
- `20260803-014210` gets a Notes line recording that the path is already
  correct, so its fix is the only thing left between that file and the gate.
  Without that line, whoever picks it up re-derives this analysis.
- The split leaves no broken intermediate state. `telegram_approval.py` is red
  today and stays red, one defect lighter.
- Proving the `OFFLINE` tuple covers every workspace member remains
  `20260804-053002`'s Done Means 3 (`test_every_package_has_a_gated_example`).
  Until it lands, enrollment is still manual and the next new example can
  repeat this rot.
