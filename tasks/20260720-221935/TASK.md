# A2: read-only agent_status (rollout-tail) + unattended /flow probe

- STATUS: OPEN
- PRIORITY: 26
- TAGS: spike,agents

## Goal

Read-only status contract + the load-bearing probe. Build
`agent_status(agent_id) -> {state, last_activity, current_tool, turns, tokens,
updated_at}` computed from the agent's codex rollout (reuse sessions.py). Design
it as a uniform contract so a detached/Claude-Code runner can fill it later.
Then PROBE the open question: run one long autonomous `codex exec` turn that
invokes /flow on a scratch project and record how it behaves unattended
(timeout, approval mode, memory growth, failure modes) before A3 commits the UI
to it.

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (Q2 rollout-tail; open question
  "does one long codex exec turn running /flow behave unattended").
- Depends on: 20260720-221929 (A1).
- Stepless direction-level task: run /plan before /work.
