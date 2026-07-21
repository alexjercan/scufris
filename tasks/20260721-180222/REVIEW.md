# Review: converge the landing + per-agent chat UI on one component

- TASK: 20260721-180222
- BRANCH: feature/converge-chat-ui

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (round-1 findings from a fresh subagent with no sight
  of the implementing session; the in-session pass re-ran both suites and
  re-derived the DoD grep + the fork-endpoint boundaries itself before adopting)

Both check suites pass: web `npm run ci` (format:check + eslint + 151 vitest +
webpack build - the build is the real type gate) and backend `python -m pytest`
(all green). The diff delivers the Goal: `agent-chat-view.ts` is now the single
chat component (`createAgentChat(root, config)` with opt-in image/slash/export/
fork), both the orchestrator landing (`startAgent`) and the project detail page
(`startAgentChat`) mount it on `#agent-chat`, and `agent-view.ts` (1263 -> 262
lines) is a pure orchestrator entry with no second chat implementation. DoD grep
`grep -rn "renderLog\|sendChatStream" web/src/agent-view.ts` -> empty. Fork
semantics split correctly and each is pinned at its own boundary (orchestrator ->
`/api/agent/session/fork` JSON new-session; project -> `/api/agents/{id}/fork` SSE
revert). Escaping of untrusted model output / user text / session titles verified.

- [x] R1.1 (NIT) tests/test_app.py:1719 - `test_agent_fork_validates` covers
  422-empty / 404-unknown / 409-orchestrator but never exercises the
  docstring-advertised "422 missing project" branch (`_require_agent_project`
  raising when `project_id` points at a deleted project). Add a case that deletes
  the project then forks and asserts 422, to pin that boundary too.
  - Response: Fixed. Added the case to `test_agent_fork_validates`: delete
    `/api/projects/my-app` then fork the orphaned `builder` agent with real text
    -> 422. Confirmed green (the 4 fork tests pass).

- [ ] R1.2 (NIT) web/src/agent-chat-view.ts `forkFrom` - if the injected
  `forkTurn` streams an error after the local `msgs` truncation, the dropped tail
  is gone from the view until a reload (only the error bubble remains). Matches
  the pre-existing landing behavior and the backend mutates nothing on a failed
  fork, so it self-heals on refresh; recorded as a known rough edge, not a defect.
  - Response: Acknowledged, left as-is. Preserves the prior landing behavior and
    the state is recoverable by reload; not worth complicating the optimistic
    truncation for a rare failed-fork path.

### Pending manual DoD items (user's to eyeball; APPROVE does not resolve these)

- manual: the landing chat looks/feels like before but is the shared component.
- manual: editing a past message on a project agent reverts that conversation in
  place in the served bundle.
