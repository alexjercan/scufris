# B2: permission modes (manual|edit|auto) replacing write_enabled, per backend

- STATUS: OPEN
- PRIORITY: 48
- TAGS: agents,backend


## Goal

Replace the `write_enabled` boolean with a Claude-style permission MODE:
`permission_mode: manual|edit|auto` (default `manual`), on `AgentRecord` +
`AgentBackend.stream(..., permission_mode=...)`, mapped per backend (codex
sandbox level; claude permission mode). Migrate persisted `write_enabled` ->
`edit` if true else `manual`.

VERIFY the exact per-backend flag for each mode LIVE (codex/claude --help + a
probe) BEFORE wiring - do not guess (lesson probe-runtime-on-target-host-early,
x3).

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (decision 3 + open question on exact flags).
- Depends on: 20260721-112429 (B1). Replaces bug #2 (write default) with modes.
