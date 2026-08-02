# Spike: in-depth logging + debug mode for scufris

- PRIORITY: 0
- TAGS: spike, observability
- KIND: SPIKE
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Research how to add in-depth, operator-readable logging (agent tool calls,
codex/CLI subprocess calls, API requests, session ops) and an easy debug mode.
Deliverable is the research doc + seeded tasks.

## Outcome

RECOMMENDED. Today logging is effectively nothing (one warning in app.py). Use
stdlib `logging` with a central `configure_logging`, a `log_level` setting, and a
`--debug`/`-v` CLI flag - zero deps, terminal-readable, uvicorn-integrated
(structlog/loguru rejected as a dep not worth it for a single-host, human-read
tool). See tasks/20260719-235458/SPIKE.md.

Seeded: 20260719-235504 (foundation: config + configure_logging + --debug +
request middleware), 20260719-235505 (instrumentation: agent/MCP/sessions logs
with redaction). Foundation first.
