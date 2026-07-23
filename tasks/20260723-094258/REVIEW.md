# Review: BC1 durable run-outcome record + AgentState.WAITING

- TASK: 20260723-094258
- BRANCH: feat/run-outcome-record

## Round 1

- VERDICT: REQUEST_CHANGES
- REVIEWER: out-of-context (round-1 findings), in-session pass verified and recorded

In-session re-verification of the load-bearing claim (R1.1): reproduced the
mid-run delete race directly - `delete("builder")` then a racing
`mark_finished("builder", ...)` writes a fresh outcome to `outcomes.json` that
survives a fresh `AgentStore` over the same `state_dir`, even though `delete`
already cleared it. Confirmed the placement asymmetry: the pre-existing
`SessionRegistry.set` sits AFTER `self._raw(agent_id)` (so it never leaks for a
deleted agent), while the new `_outcomes.set` was placed BEFORE the split. Full
suite `344 passed`, ruff + mypy clean before the fix.

- [ ] R1.1 (MAJOR) scufris/agent_store.py `mark_finished` - the
  `self._outcomes.set(...)` write is placed ABOVE the regular-agent
  `self._raw(agent_id)` existence check, so an agent deleted mid-run (an
  anticipated path - `app.py` persist comment: "if the agent was deleted mid-run,
  mark_finished raises AgentNotFound, which the supervisor swallows") gets a fresh
  outcome resurrected that survives restart. This defeats Step 5 / DoD ("Deleting
  an agent removes its outcome entry") and the `delete` comment's own claim that a
  reused id "can never inherit ... a stale needs input outcome". The sibling
  `SessionRegistry.set` is placed AFTER `_raw`, so it does not leak; the new write
  is inconsistent with it and worse. Fix: write the outcome only once existence is
  established - in the orchestrator branch, and after `_raw(agent_id)` for a
  regular agent (mirroring where the session id is set).
  - Response: fixed this round. `mark_finished` now BUILDS the `RunOutcome` up
    front but WRITES it (`self._outcomes.set`) only after existence is
    established - in the orchestrator branch, and after `_raw(agent_id)` for a
    regular agent, alongside where the session id is set. A delete-racing
    completion now raises `AgentNotFound` before any write. Pinned by the new
    `test_delete_then_mark_finished_does_not_resurrect_outcome` (R1.2), which
    failed red on the pre-fix code (`assert RunOutcome(...state=DONE,
    message='late'...) is None`) and is green after.
- [ ] R1.2 (MINOR) tests/test_agent_store.py - no coverage for the
  delete-then-`mark_finished` race that actually leaks; `test_delete_removes_outcome`
  only covers a straight delete. Add a regression test that deletes an agent and
  then calls `mark_finished` (expecting `AgentNotFound` and no resurrected
  outcome), which would have caught R1.1 and guards it.
  - Response: fixed this round. Added
    `test_delete_then_mark_finished_does_not_resurrect_outcome` - deletes the
    agent, expects `AgentNotFound` on the racing `mark_finished`, and asserts no
    outcome in-process and after a restart. Was red pre-fix (reproduced R1.1),
    green post-fix.
- [ ] R1.3 (MINOR) tests/test_agent_store.py - the error-terminal-state outcome
  path has no test. On an error turn the backend yields `StreamError` (not
  `StreamDone`), so `message` stays `""` and `mark_finished` records
  `state=ERROR, message=""`. Correct behavior, but untested through the outcome
  store; add a cheap store-level test.
  - Response: fixed this round. Added `test_error_terminal_outcome_recorded`
    (`mark_finished(state=ERROR)` records an ERROR outcome with `message == ""`,
    no crash).
- [ ] R1.4 (NIT) scufris/agent_store.py - `RunOutcome.message` stores the full
  final reply uncapped (acknowledged in the close record). Fine for v1 (one entry
  per agent); noted only because a long message is persisted verbatim each run.
  - Response: acknowledged, left as-is for v1 (one entry per agent, small). A cap
    can be added when/if the message ever bloats; noted in the close record.

No open `manual:` DoD items for this task (all proofs are `test:`/`cmd:`).

## Round 2

- VERDICT: APPROVE
- REVIEWER: out-of-context (resumed against the R1.1 fix), in-session pass verified

R1.1 re-reproduced by the reviewer: `delete("builder")` then a racing
`mark_finished("builder", ...)` now raises `AgentNotFound` and leaves NO outcome,
in-process and after a fresh `AgentStore` over the same `state_dir`; the fix holds
for both branches. R1.2's regression test confirmed to fail under the pre-fix
ordering and pass with the fix. Full suite `346 passed`, ruff + mypy clean.

- [x] R1.1 - RESOLVED. Outcome written only after existence is established
  (orchestrator branch; after `_raw` for a regular agent).
- [x] R1.2 - RESOLVED. `test_delete_then_mark_finished_does_not_resurrect_outcome`
  pins the fix (red under the old ordering, green now).
- [x] R1.3 - RESOLVED. `test_error_terminal_outcome_recorded` covers the
  ERROR/empty-message path.
- [x] R1.4 - ACKNOWLEDGED, left as-is for v1.

- [x] R2.1 (NIT) tasks/20260723-094258/REVIEW.md - a stray literal `</content>`
  line at end of file (a Write-tool artifact); deleted this round.
  - Response: removed.
