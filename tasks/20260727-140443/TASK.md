# surface run.error / backend StreamError detail through agent_status and pending_agents

- STATUS: OPEN
- PRIORITY: 1
- TAGS: backlog,agent,orchestrator,dx

## Story

As an operator watching a delegated sub-agent, I want `agent_status` /
`pending_agents` to report WHY a run ended in error, so a backend `StreamError`
(idle timeout, over-limit line, thread-setup failure) shows a diagnostic message
instead of an empty one.

## Context

Split out of `20260727-133302` (>64 KiB app-server line fix). A backend-level
`StreamError` is published to the run bus and the run ends in `DONE` state
(`supervisor.py` `_drain` only publishes events; it does not set `run.error` for
a StreamError event). The orchestrator's `agent_status` reports `read_status`
fields, not the last `StreamError`, so any backend error (not just the
over-limit one) surfaces to the orchestrator as "error with no message". This is
a general orchestrator-visibility gap, independent of any single backend bug.

## Steps

- [ ] Decide the surface: have `_drain` record the last `StreamError.detail` onto
      `run.error` (without forcing ERROR state), or have `agent_status` /
      `pending_agents` read the terminal StreamError from the run bus/state.
- [ ] Thread that detail through `agent_status` and `pending_agents` output.
- [ ] Test: a run whose backend yields a `StreamError` reports the detail via
      `agent_status`.

## Notes

- Files: `scufris/supervisor.py` (`_drain` ~318, error handling ~277-295),
  the orchestrator status tool (`agent_status` / `pending_agents`).
- Affects every backend `StreamError` equally (idle-timeout path
  `agent.py` included), so fixing it here helps all error paths, not just the
  over-limit line.
