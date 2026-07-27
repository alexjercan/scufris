# Review: report_back sub-agent completion tool

Reviewed the uncommitted working-tree change on branch `feat/report-back-tool`
against the task spec (TASK.md / DECISION.md). The feature adds an
`AgentState.REPORTED` outcome plus a `report_back(summary)` sub-agent callback
tool mirroring `request_input`, wired through the store, app endpoint, wake
bridge, MCP surface, steering preamble, and a CSS badge.

Verification run in the nix devshell: `python -m pytest` (relevant suites, 257
tests) PASS; `ruff check .` PASS; `mypy scufris/` PASS (25 files, no issues).

Overall the implementation is a faithful, careful sibling of `request_input`.
The preserve generalization is correct, REPORTED is threaded through every
consumer I could find (pending_outcomes, wake filter, wake_prompt, pending
endpoint, MCP rendering), the role scoping is right (agent-only, not
orchestrator), the wake_prompt signature change has all callers/tests updated,
and the tests exercise the real behavior including the preserve-through-DONE,
error-wins, stale-overwrite, and auto_wake-off invariants. No blocking issues.
The findings below are a base-branch hazard and documentation gaps.

### [MAJOR] Branch is based on a stale master; `git diff master` shows spurious frontend reverts

The worktree HEAD is `a0350e7`, but `master` has advanced two commits past it
(`ca0a29d` "render sub-agent Tools as orchestrator-style cards" and `e4efa76`
LESSONS). Because of that drift, `git diff master -- .` reports changes to
`LESSONS.md`, `web/src/agent-settings-view.ts`, `web/src/agent-settings-view.test.ts`,
and `web/src/settings-view.ts` that are NOT part of this feature - they are the
diff engine showing master's newer commits as deletions/reverts. The actual
feature change (`git diff HEAD`) touches only: enums, agent_store, app, wake,
mcp_server, sessions, tests, and style.css - all clean.

Why it matters: if this branch is merged or rebased as-is, it will REVERT
master's `toolCard`-export / sub-agent-tools-card work and drop a LESSONS entry.
That is a real regression risk, not just review noise.

Suggested change: rebase `feat/report-back-tool` onto current `master` before
landing (`git rebase master`), then re-verify the web build/tests, and confirm
`git diff master` shows only the report_back files. The report_back change itself
does not conflict with the frontend work (disjoint files), so the rebase should
be clean.

### [MINOR] `GET /api/agents/pending` handler docstring not updated for REPORTED

scufris/app.py:1123-1125. The `PendingAgent` model comment (app.py:485-489) was
correctly updated to mention REPORTED, but the endpoint docstring that produces
those rows still reads "unacknowledged needs-input (WAITING, from request_input)
or ERROR outcome ... to find blocked sub-agents". It omits the REPORTED /
finished case the endpoint now returns.

Suggested change: extend the first sentence to include REPORTED, e.g.
"unacknowledged needs-input (WAITING), reported-done (REPORTED), or ERROR
outcome ... to find blocked or finished sub-agents", matching the wording already
used in `pending_outcomes()` and the `PendingAgent` model comment.

### [MINOR] MCP `acknowledge()` docstring omits the reported case

scufris/mcp_server.py:634-636. The tool docstring says to call it "after you have
answered its `request_input` question or dealt with its error" - it never mentions
acknowledging a `reported` agent, which is now the primary post-`report_back`
action and is the exact instruction `pending_agents()` and `wake_prompt` steer the
orchestrator toward ("read its report ... then acknowledge"). A reader keying off
this docstring alone would not learn REPORTED is an ack-able state.

Suggested change: add the reported case, e.g. "... answered its `request_input`
question, read its `report_back` result, or dealt with its error."

### [NIT] `mark_finished` preserve comment says "delete-mid-run cannot resurrect it" but only mark_finished guards existence, not report_back

scufris/agent_store.py:743-751 (the preserve comment) and report_back
(agent_store.py:833-843). This is not a bug - `report_back()` does call
`self._raw(agent_id)` as an existence guard before writing (agent_store.py:838),
exactly like `request_input`, so a deleted agent raises AgentNotFound and writes
nothing (covered by `test_report_back_on_deleted_agent_raises`). The nit is only
that the long preserve comment reasons about the write ordering of the WAITING/
REPORTED signal vs the DONE without restating that the signal write itself is
guarded; a one-line note that report_back mirrors request_input's existence guard
would make the invariant self-evident. Optional.

### [NIT] `report_back` store method does not carry `backend` like `mark_finished`

scufris/agent_store.py:833-843. `report_back()` records the outcome with
`session_id` but no `backend`, matching `request_input()` exactly (which also
omits it) - the backend is stamped later by the turn-end `mark_finished` via the
registry. This is intentional and correct (consistent with the sibling), noted
only to confirm it was checked, not missed.

## Consumer sweep (no gaps found)

Checked every consumer of the AgentState enum for a WAITING/ERROR/DONE branch
that should also include REPORTED:

- `pending_outcomes()` (agent_store.py:867-872): includes REPORTED. OK.
- `mark_finished` preserve (agent_store.py:753-770): WAITING-or-REPORTED,
  `eff_state = existing.state`; ERROR and later-run DONE still overwrite. Correct
  and test-covered (`test_error_after_report_back_wins`,
  `test_stale_reported_overwritten_by_a_new_run`,
  `test_reported_survives_same_run_completion`).
- `WakeBridge.on_run_complete` filter + `wake_prompt` (wake.py:84-95, 32-49):
  REPORTED enqueued and rendered with tailored read+ack guidance. Signature
  changed `dict[str,str]` -> `dict[str,tuple[AgentState,str]]`; the only caller
  is the internal `_drain`, and the only external constructor is the test, both
  updated.
- app.py `/api/agents/pending` (1134-1158) and `agent_run_status` (1487-1516):
  pass `o.state` / run-state through generically; no hard-coded state set.
- MCP `pending_agents()` rendering (mcp_server.py:625-628): renders `state`
  generically at width 8 ("reported" is 8 chars, fits exactly).
- Frontend: badge class is built dynamically as `agents__badge--${state}`
  (web/src/agents-view.ts:52, agent-detail-view.ts:28) and the `state` field is
  typed `string` (common.ts:422,432), not a union - the new `.agents__badge--
  reported` rule (style.css:2055-2059) is picked up with no TS change. Frontend
  `state ===` checks only compare against `"idle"`, unaffected.
- telegram.py and health.py do not reference AgentState. No serialization,
  OpenAPI-enum, or exhaustive-match break: `AgentState` is a `StrEnum`, so adding
  a member is additive for pydantic/OpenAPI and no `match`/exhaustive switch over
  it exists.

Role scoping confirmed: `_AGENT_ROLE_TOOLS = {"request_input", "report_back"}`
(mcp_server.py:933); `apply_role(ROLE_AGENT)` keeps exactly those two,
`apply_role(ROLE_ORCHESTRATOR)` removes exactly `["report_back","request_input"]`
and the orchestrator STEERING_PREAMBLE gains no `report_back` - all asserted in
tests. `AGENT_STEERING_PREAMBLE` stays one `[scufris-tools]` block with the added
finish clause; `strip_steering` round-trip is covered by the existing single-block
test.

VERDICT: REQUEST_CHANGES

---

## Round 1 resolution (author)

- [MAJOR] base drift: resolved by merging current `master` into the branch (the
  flow landing procedure - master had advanced to `e4efa76` after this branch was
  sprouted from `a0350e7`, a concurrent-session move, cf. lesson
  `recheck-head-before-committing-in-a-user-touched-repo`). Conflicts resolved on
  the branch, full QA gate re-run green, and `git diff master` now shows only the
  report_back files. The frontend work (`agent-settings-view.ts`,
  `settings-view.ts`) and the LESSONS entry are preserved (disjoint files).
- [MINOR] `GET /api/agents/pending` docstring: updated to name REPORTED
  (app.py:1123-1128).
- [MINOR] MCP `acknowledge()` docstring: updated to name the report_back result
  case (mcp_server.py:634-637).
- [NIT] preserve-comment / backend-omission: intentional and consistent with the
  `request_input` sibling; the existence guard (`_raw`) in `report_back` mirrors
  `request_input` and is covered by `test_report_back_on_deleted_agent_raises`.
  Left as-is by design.

## Round 2 verdict

Feature diff clean and confirmed by the reviewer's full consumer sweep; all
actionable findings addressed; branch brought up to date with master and
re-verified green.

VERDICT: APPROVE
