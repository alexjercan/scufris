# Retro: report_back sub-agent completion tool

## What went well

- The feature was a near-exact sibling of `request_input`, so the strongest
  move was to trace that one signal end-to-end FIRST (two parallel Explore
  agents: one for the MCP/agent side, one for the backend/store/wake/tests
  side) and then mirror it layer by layer. Every layer had an existing pattern
  to copy, which kept the change small and consistent and made the tests
  obvious (each new test is a sibling of a WAITING test).
- Confirming the two load-bearing forks up front (new `AgentState.REPORTED` vs
  DONE+flag; wake mirrors `request_input` vs force-wake) with an
  `AskUserQuestion` before writing the plan meant zero rework: the whole build
  followed from those two answers. Recorded in DECISION.md.
- Generalizing `preserve_waiting` to `preserve_signal` (WAITING **or**
  REPORTED, `eff_state = existing.state`) was the one genuinely non-mechanical
  spot, and pinning it with three tests (preserve-through-DONE, error-wins,
  stale-overwrite-by-later-run) matched exactly the invariants the WAITING path
  already had - no new invariant invented.
- A proactive consumer sweep (grep every `AgentState.WAITING`/`ERROR` reader)
  BEFORE the out-of-context review caught two doc-accuracy misses the reviewer
  would otherwise have flagged (config.py `auto_wake` comment, and the
  CHANGELOG), and confirmed the frontend badge is dynamic (`state: string`, not
  a union) so no TS change was needed.

## What went wrong / difficulties

- Base drift: master advanced two commits (to `e4efa76`) AFTER this branch was
  sprouted from `a0350e7`, because a concurrent session moved it. `git diff
  master` then showed spurious "reverts" of unrelated frontend + LESSONS files.
  The out-of-context reviewer correctly flagged this as a MAJOR merge hazard.
  It was not a code defect - `git diff HEAD` was clean - but it is exactly the
  `recheck-head-before-committing-in-a-user-touched-repo` situation, and the fix
  is the flow's own update-from-master landing step (merge master into the
  branch, resolve, re-verify) before squash-landing.
- `wake_prompt`'s signature had to change (`dict[str,str]` ->
  `dict[str,tuple[AgentState,str]]`) to label each agent's state in the wake
  prompt. That is a real API change, so its one caller (`_drain`) and its one
  test constructor both had to move in lockstep - caught by remembering the
  `signature-change-breaks-test-doubles` lesson, not by the type checker alone
  (the test built the dict literally).
- `ruff format` merged an edited docstring sentence onto an over-long comment
  line in mcp_server.py; had to re-wrap it by hand after formatting. Re-reading
  the produced text (not trusting the format success) caught it.

## What to improve next time

- When a task will `git diff <default>` for review, note the sprout base commit
  up front and, if a concurrent session may be moving master, merge master into
  the branch BEFORE requesting the out-of-context review - so the reviewer sees
  only the feature diff and does not spend a finding on base drift. (The review
  still would have passed, but the MAJOR was avoidable noise.)
- For a "mirror an existing tool" feature, the parallel two-Explore-agents +
  layer-by-layer-sibling approach worked so well it is worth naming as the
  default playbook: map the reference signal exhaustively, then copy it, then
  copy its tests.
