# Refactor: use enums/Pydantic for stringly-typed options (auth_mode, backend, permission_mode, etc.)

- STATUS: CLOSED
- PRIORITY: 20
- TAGS: refactor, backend
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

User feedback (2026-07-21): "add a task to try and use more enums and be more
clean (maybe even using Pydantic) e.g. when doing `auth_mode: str` maybe using
Enum; and having explicit exhaustive options in most places; but it still works
now, so that can be a refactor code improvements task." Low priority - a
cleanliness/maintainability pass, not a behavior change.

## Steps

- [x] Inventory the stringly-typed option fields/params across the codebase.
      Done - see the "Converted / deliberately left" notes below.
- [x] Introduce `enum.StrEnum`s for the fixed option sets in a central
      `scufris/enums.py`: `AuthMode`, `Backend`, `PermissionMode`, `AgentState`,
      `RunPhase`. StrEnum so members equal their string (no wire change) and
      pydantic validates membership.
- [x] Wire them in: config (`agent_auth_mode`, `agent_backend`), agent_store
      (`AgentRecord.state`/`permission_mode`, orchestrator state, mark_*),
      supervisor (`RunState.state` - was bare `RunLifecycle = str`), app models
      (auth_mode / permission_mode / agent_backend fields). `PERMISSION_MODES`
      derives from the enum; `normalize_permission_mode` returns it.
- [x] Behavior-preserving: no API/wire change (the full pre-existing suite passes
      UNCHANGED), legacy `app_server|exec -> codex` coercion still loads. Added
      `tests/test_enums.py` pinning membership validation + the string round-trip;
      coerced `mark_finished` so a `model_copy(update=...)`-set str state still
      lands as the enum (else pydantic warns on serialize).

## Definition of Done

- `auth_mode` (and the other identified bare-`str` option fields) are enums/
  Literals with exhaustive options, validated by pydantic
  (test: an invalid value is rejected; a valid one round-trips unchanged on the
  wire).
- No behavior/wire change: existing tests pass unchanged (or only for stricter
  validation) (cmd: full suite green).
- Notes record which fields were converted and which were deliberately left.

## Converted / deliberately left

CONVERTED to `StrEnum` (scufris/enums.py):
- `AuthMode` <- config.agent_auth_mode; app AgentInfo/AccountInfo/AgentConfig.auth_mode.
- `Backend` <- config.agent_backend; app AgentConfigUpdate.agent_backend (legacy
  app_server|exec still fold to codex via the config validator/canonical_backend).
- `PermissionMode` <- AgentRecord.permission_mode; app AgentCreate/AgentUpdate;
  config.normalize_permission_mode returns it; PERMISSION_MODES derives from it.
- `AgentState` <- AgentRecord.state + orchestrator state + mark_running/mark_finished
  (`AgentLifecycle` kept as an alias of it for existing importers).
- `RunPhase` <- supervisor RunState.state + the run-engine state assignments
  (this was the only truly UNtyped one: `RunLifecycle = str`).

LEFT as `str` (on purpose):
- `ToolCall.status` (sessions.py): backend-defined, open-ended tool statuses - not
  a fixed set to enumerate.
- `AgentRunStatus.state` (app.py): a MERGED display value (RunPhase OR AgentState -
  their union), so one enum would be lossy; a StrEnum still serializes fine.
- backends.py `permission_mode` params: input-boundary str that map via dicts keyed
  by the value; the value passed is always a PermissionMode (a str subclass).
- `config.log_level`: the logging module accepts a broader set (CRITICAL/NOTSET).
- `AgentRecord.backend`: holds canonical_backend() output (can fall through to an
  arbitrary folded string); creation is gated by available_backends.
- Already `Literal`/typed and left: session `action`, StreamEvent `kind`s.

## Notes
- Priority 20 (LOW) - do LAST, after the feature/bug work. Pure code-quality.
- Keep `StrEnum` (not plain `Enum`) so JSON stays the same string values.
