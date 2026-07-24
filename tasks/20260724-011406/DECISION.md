# Decision: runner timeout is an idle guard, not a per-turn wall-clock

- STATUS: ACCEPTED
- DATE: 20260724
- UMBRELLA: 20260724-081616

## Context

`_stream_app_server` (`scufris/agent.py`) enforced a single per-turn deadline
`now + agent_timeout_seconds` (120s) over the whole turn, killing any turn that
ran past 120s wall-clock even while actively streaming. This defeated the
ADR-001 supervisor (`scufris/supervisor.py`), which had already replaced the
wall-clock request timeout with a no-output stall guard
(`agent_heartbeat_seconds`, `budget_seconds=None`).

## Decision

Repurpose `agent_timeout_seconds` as a runner IDLE (inter-line) timeout: each
app-server `readline` gets `timeout=agent_timeout_seconds`; the deadline is
effectively reset on every line that returns data. A turn times out only after
that many seconds of SILENCE from the app-server, never for total length.

## Alternatives considered

- Remove the runner cap entirely and rely solely on the supervisor's 600s
  heartbeat. Rejected: the runner is also used standalone (`cli.py`) with no
  supervisor, so it needs its own liveness guard.
- Add auto-retry so a stalled/failed turn transparently continues. Deferred:
  with an idle guard a progressing turn never times out, so the reported bug is
  fixed without retry. Retry is a larger design concern spiked separately
  (task 20260724-081811).

## Consequences

- A long-but-progressing turn (long conversation turn, spawned sub-agent doing
  minutes of work) runs to completion.
- A genuinely hung app-server (no output past the idle bound) is still cut -
  stall guard preserved.
- `agent_timeout_seconds` changes meaning from "per-turn wall-clock" to
  "max silence between app-server lines"; its docstring is updated to say so.
