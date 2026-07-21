# Refactor: use enums/Pydantic for stringly-typed options (auth_mode, backend, permission_mode, etc.)

- STATUS: OPEN
- PRIORITY: 20
- TAGS: refactor,backend

## Story

User feedback (2026-07-21): "add a task to try and use more enums and be more
clean (maybe even using Pydantic) e.g. when doing `auth_mode: str` maybe using
Enum; and having explicit exhaustive options in most places; but it still works
now, so that can be a refactor code improvements task." Low priority - a
cleanliness/maintainability pass, not a behavior change.

## Steps

- [ ] Inventory the stringly-typed option fields/params across the codebase:
      `auth_mode`, backend ids/modes, `permission_mode`, agent `state`
      (AgentLifecycle), tool-call `status`, SSE event `kind`, etc. Note which are
      already `Literal[...]` (keep those or promote to `StrEnum` for shared reuse)
      and which are bare `str`.
- [ ] Introduce `enum.StrEnum`s (or shared `Literal` aliases) for the ones that
      are bare `str` but have a fixed, known set - starting with `auth_mode`.
      Prefer `StrEnum` so values still serialize as their string (no wire-format
      change) and pydantic validates membership.
- [ ] Replace exhaustive `if/elif` string chains with enum-driven maps where it
      reads cleaner (backend/permission mappings already use dicts - keep).
      Ensure exhaustiveness where a missing case would be a bug.
- [ ] Keep it behavior-preserving: no API/wire changes (StrEnum members equal
      their string), migrations for any persisted value stay valid. Full suite
      green before/after.

## Definition of Done

- `auth_mode` (and the other identified bare-`str` option fields) are enums/
  Literals with exhaustive options, validated by pydantic
  (test: an invalid value is rejected; a valid one round-trips unchanged on the
  wire).
- No behavior/wire change: existing tests pass unchanged (or only for stricter
  validation) (cmd: full suite green).
- Notes record which fields were converted and which were deliberately left.

## Notes
- Priority 20 (LOW) - do LAST, after the feature/bug work. Pure code-quality.
- Relevant: scufris/config.py (auth_mode, agent_backend), scufris/agent_store.py
  (AgentLifecycle, PermissionMode), scufris/agent.py (Stream* kinds, ToolCall
  status), scufris/backends.py (CodexMode if it survives the exec-drop task).
- Keep `StrEnum` (not plain `Enum`) so JSON stays the same string values.
