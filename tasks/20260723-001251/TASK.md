# Persisted agent<->session id registry (fix orchestrator/sub-agent session mixing)

- STATUS: OPEN
- PRIORITY: 42
- TAGS: bug,agents,backend,sessions

## Story

As the operator, I want each agent's (including the orchestrator's) codex/claude
session tracked by an explicit, persisted `(agent_id -> session_id)` mapping, so the
orchestrator's conversation never gets mixed up with a sub-agent's when both run on
codex.

## Context (grounded / root cause)

The orchestrator's session id is IN-MEMORY only (`agent_store.py:112`,
`self._orch_session_id`, set/read via `set_orchestrator_session` /
`orchestrator_session_id`) and is NOT persisted - it resets on restart. Sub-agents
persist `session_id` on their `AgentRecord` (`agents.json`). Both share one
`CODEX_HOME`. When a read path (transcript/status) resolves "the session" loosely,
or the orchestrator's in-memory id is `None` after a restart, it can latch onto the
most-recent codex rollout - which may be a sub-agent's. Same-backend (both codex)
makes them indistinguishable to a loose resolver, which is the mixing observed.

## Direction

Make the session<->agent mapping a first-class, persisted structure for ALL agents
(orchestrator included), and resolve sessions ONLY through it:

- [ ] A persisted registry of `(agent_id -> current session_id [+ backend])`, so a
      codex id and a claude id for the same agent never cross. The orchestrator uses
      this store instead of the in-memory `_orch_session_id` field.
- [ ] Every turn resolves/writes the session via the registry keyed by `agent_id` -
      never by "latest rollout" or a cwd-hash guess.
- [ ] On agent delete, remove its mapping (and optionally its rollout). On backend
      switch, clear the stale cross-backend id.
- [ ] Reproduce FIRST: an orchestrator turn + a codex sub-agent turn must keep
      distinct sessions/transcripts, including across a simulated restart (the
      in-memory id being lost is part of the bug).

## Definition of Done

- Orchestrator and a codex sub-agent never share a session/transcript. (test:
  reproduction of the mixing, red before the fix)
- Deleting an agent removes its session mapping. (test)
- Switching an agent's backend clears the stale id instead of resuming a
  wrong-backend session. (test)

## Notes

- Folds/relates to open bugs: `20260721-152034` (stale cross-backend session; claude
  resume fails) and `20260720-020345` (list app_server sessions / originator fix) -
  consider subsuming or closing them.
- Lesson context (`LESSONS.md`): `backends-tag-provenance-differently` (codex exec
  vs app-server originator); `probe-the-stateful-path-not-the-one-shot`.
- Side benefit: persisting the orchestrator's session survives a restart (today its
  conversation is lost).
