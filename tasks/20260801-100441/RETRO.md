# Retro: Extract the orchestrator-turn and agent-run services

- TASK: 20260801-100441
- BRANCH: refactor/orchestrator-services
- REVIEW ROUNDS: 2

## What went well

- Planning found the real seam before any code moved: `_launch_agent_turn` was
  already the single funnel for all six callers, so the task was a move plus a
  state relocation, not a rewrite. Steps needed no restructuring mid-work.
- Planning also falsified its own DoD proof. The original
  `! rg -n "fastapi|telegram" scufris/services/` was green on base for the wrong
  reason (`rg` on a missing directory exits non-zero). The `test -d && !` guard
  replaced it and is red on base AND red if the directory is later emptied.
- The SSE characterization was checked against `master` byte-for-byte rather
  than against its own capture, so "unchanged" meant unchanged.
- Round-1 fixes were falsified by mutation, not by reading: inverting `wake`'s
  `RunAlreadyActive` branch and `cancel`'s `NoActiveRun` branch reds each new
  assertion.

## What went wrong

- Seven tests went green-to-red mid-work with live token counts.
  `monkeypatch.setattr("scufris.app.get_backend", ...)` stopped intercepting
  once the resolve moved into `orchestrator/runs.py`. Patching a name where it
  is USED is correct Python; the failure mode is that a fake which stops being
  installed looks exactly like a fake that is working. Centralised both bind
  sites in `conftest.patch_get_backend`.
- The DoD named `npm run test:e2e`, a script that never existed on this branch
  or `master`. Planning anticipated the browser suite being unrunnable but not
  absent - it assumed a harness from the shape of the repo instead of reading
  `web/package.json`. Corrected to `npm run ci`.
- Two invariants were dropped rather than moved. Round 1 caught `turn.cancel`'s
  idle-False (its old test became a fake echoing its own constructor argument)
  and `wake`'s `RunAlreadyActive` back-off. Both are branches a REPLACED test
  used to pin, which is the class of loss a mechanical move hides.
- Two service methods (`require_agent_async`, `require_agent_project_async`)
  were extracted with zero callers - speculative surface, caught at R1.1.
- The branch appended a 40-line diagnosis to a sibling record
  (`tasks/20260803-043935/TASK.md`). Accurate and useful, but scope the Story
  did not name; the reviewer logged it as a process signal.

## What to improve next time

- Breadth: the diff is large (~2000 insertions) but honestly so - one funnel
  function plus its state, and 489 of those lines are the new test module.
  `app.py` shrank 2923 -> 2621. No independently landable split was missed.
- Churn: all nine round-1 findings were "the move dropped something", not "the
  design is wrong". The plan-time question that would have prevented most of
  them: for each test being REPLACED, name the invariant it pins and where the
  replacement pins it. Six of nine findings fall out of that one list.
- Context: no compaction warning or handoff occurred; the work ran in one
  context. The only measured pressure was the 600-line source cap, which is why
  turn/runs/errors are three modules - `runs.py` sits at 499.

## Action items

- Follow-up worth its own task: `/api/chat/stream` leaks the image tempdir when
  the launch refuses (`cleanup` is wired only as `on_done`, which never fires if
  `turn.stream` raises). Pre-existing on `master`, out of this diff's scope.
- R2.1 (NIT, non-blocking) fixed during close-out: the two ragged comment wraps
  left by the round-1 edits are rewrapped to 88 columns. Comment text only;
  ruff, ruff format and mypy re-run clean.
- R1.9 stands acknowledged-no-change:
  `test_host_action_api.py::test_cancelling_a_live_apply_is_recorded` was seen
  red once by the out-of-context reviewer and never reproduced. Not added to
  the flake list; pre-blaming an unreproducible flake is how a regression gets
  waved through.
