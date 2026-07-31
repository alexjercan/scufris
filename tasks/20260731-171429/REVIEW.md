# Review: Split the Telegram surface under the size cap

- TASK: 20260731-171429
- BRANCH: refactor/split-telegram-surface
- BASE: master (a74c4e8)

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context subagent, prompt limited to task ID, worktree,
  branch, dimensions and record format.
- The primary re-ran every check and independently re-derived three
  load-bearing claims before accepting the report: no submodule imports its own
  package `__init__` (grep, none); every name in the facade `__all__` has a
  consumer in `app.py`, `examples/` or the two test files (per-name grep, none
  unconsumed); and `ruff format --check` on the merge-base
  `scufris/telegram.py` is red, so the drift on the moved strings is inherited.

Clean on correctness, spec, tests and design. The reviewer read the merge-base
file against all eight modules by hand rather than trusting the suite, and
confirmed every restructured site preserves behavior: the guard order in
`_handle_update`, the `handle_reason_reply` True/False inversion (including the
"a command is not a reason" fall-through), the whole `handle_callback` decision
path, the throttle-clock semantics in `_render_turn`, and the unguarded plain
resend inside `send_reply`. One benign observable delta: log records now carry
`scufris.telegram.<submodule>` as the logger name. `logsetup` sets the level on
the `scufris` root, so filtering and message text are unaffected.

All 12 Steps and all 8 Definition of Done clauses proven. Baseline of 896 tests
independently reconfirmed at the merge-base.

### Findings

| # | Severity | Where | Finding | Response |
|-|-|-|-|-|
| R1.1 | NIT | tasks/20260731-171429/TASK.md close-out | The close-out claims exactly two plan corrections; there is a third. The Steps put "the typing loop" in `turn.py`, but `_try_typing` landed as `BotApi.try_typing`. | Accepted. Third row added to the corrections table, naming the split (delivery policy vs turn lifetime) and that review found it. |
| R1.2 | NIT | scufris/telegram/turn.py:46 | `keep_typing` lost its underscore but has no caller outside the module; `bot.py` imports only `drive_turn` and `DEFAULT_EDIT_INTERVAL`. | Accepted, and applied to `render_turn` as well, which has the same single in-module caller. Both are now `_keep_typing` / `_render_turn`; the module docstring follows. |
| R1.3 | NIT | scufris/telegram/text.py:23 | `_ELIDED` is read only by `render.render_approval`, and `text.py`'s own `MAX_MESSAGE` comment points the reader at that function to explain it. | Accepted. Moved to `render.py` beside its single use, with a comment naming what it replaces. One cross-module private import gone. |
| R1.4 | NIT | scufris/telegram/__init__.py:3 | The facade docstring says "the three things the surface does" over a seven-row table. | Accepted. Reworded to "transport, command handling, and rendering", which is the task's own phrasing for the seams. |

No BLOCKER or MAJOR. All four NITs accepted and fixed on the branch before
landing; re-verified after the fixes: `ruff check` and `ruff format --check`
clean, `mypy` clean on 88 files, `pytest` 896 passed, `check_file_size.py`
green.

### Pending manual items

- The epic's manual acceptance ("opening any single component for a change
  requires reading files that fit in one implementation context") stays pending
  until every child lands. The largest file in this package is 472 lines.

## Verdict

- VERDICT: APPROVE
