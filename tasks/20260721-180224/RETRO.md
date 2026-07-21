# Retro: retire the codex-exec runner + fix the settings-view backend picker

- TASK: 20260721-180224
- BRANCH: feature/retire-exec-runner (landed c5bc8e7)
- REVIEW ROUNDS: 1 (APPROVE, out-of-context)

See TASK.md for what/why and REVIEW.md for the findings. Process notes only here.

## What went well

- Mapped the call graph before deleting: counted every helper's usages in
  agent.py and traced which were exec-only (`_parse_events`, `_tool_call_from`,
  `_usage_from`, `_exec_args`, the two runners) vs shared with the app-server path
  (`_mcp_overrides`, `_steer`, `_parse_event_line`, `_turn_mode`). So the deletion
  left no dead code and did not nick the survivor - the reviewer confirmed zero
  dangling references.
- Preserved coverage by RE-POINTING, not just deleting: the exec tests also held
  shared behavior (missing-binary guard, the cwd seam, image attach), so those
  were re-pointed onto `_stream_app_server` with a real fake app-server instead of
  dropped. Coverage survived the retirement.
- Stopped and asked when the DoD ("picker shows Codex/Claude") conflicted with the
  actual model (the orchestrator's `agent_backend` could only be app_server|mock,
  never Claude). That was a real fork, not a detail; the user's answer (widen the
  vocab + gate mock behind the dev flag) shaped the correct design rather than a
  guess that would have shipped a Codex/Mock picker.

## What went wrong

- The task's "low-complexity cleanup" framing was wrong: it grew into a config
  schema widen (`agent_backend` -> codex|claude|mock) touching health. Root cause:
  the task was seeded before the backend model was pinned, and its DoD asserted a
  user-facing capability (run the orchestrator on Claude) that did not yet exist -
  so "make the picker say Claude" was really "make Claude selectable".
- Two review NITs: a health-test docstring described the unset-binary path while
  the test exercised the broken-binary path; and I did not proactively pin the
  strict-input contract (raw `app_server` PATCH -> 422) even though the task text
  itself flagged "API input stays strict" as load-bearing.

## What to improve next time

- When a "cleanup" task's DoD asserts a user-facing CAPABILITY, verify the
  capability exists in the current model before assuming the task is trivial - if
  it doesn't, the task is a feature, and surface that up front.
- When the task text names a contract as load-bearing (here strict input reject),
  write its regression test in the first pass, not after review asks.

## Action items

- [x] Adopted R1.1 (docstring) + R1.2 (strict-reject regression test).
- No follow-up tasks: B5e is the last B5 sub-task; the manual DoD (eyeball the
  picker + switch the orchestrator to Claude) batches to the goal's Finish.
