# B2: permission modes (manual|edit|auto) replacing write_enabled, per backend

- PRIORITY: 48
- TAGS: agents, backend
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

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

## Steps

Flags VERIFIED LIVE (codex exec --help / claude --help):
- codex --sandbox: read-only | workspace-write | danger-full-access
- claude --permission-mode: default | acceptEdits | bypassPermissions | plan | auto
Mapping (default `manual`):
  manual -> codex read-only,        claude default
  edit   -> codex workspace-write,  claude acceptEdits
  auto   -> codex danger-full-access, claude bypassPermissions

- [x] config.py: `PERMISSION_MODES = ("manual","edit","auto")`; a
      `normalize_permission_mode(m)` (unknown -> "manual").
- [x] backends.py: replace `stream(..., write_enabled=bool)` with
      `permission_mode: str = "manual"` on the protocol + all backends;
      `_codex_sandbox_for(mode)` / `_claude_permission_mode_for(mode)` maps;
      CodexBackend passes the sandbox, ClaudeBackend always passes
      `--permission-mode <mapped>`.
- [x] agent_store.py: `AgentRecord.permission_mode: PermissionMode = "manual"`
      replaces `write_enabled`; create/update take `permission_mode` (validated);
      MIGRATE on load: legacy `write_enabled` -> `edit` if true else `manual`.
- [x] app.py: `AgentCreate`/`AgentUpdate` `permission_mode` (Literal) replaces
      `write_enabled`; run/chat pass `permission_mode=agent.permission_mode`.
- [x] mcp_server.py: `_agent_status_text` shows `mode: <permission_mode>` instead
      of `writes: ...`.
- [x] common.ts + agents-view.ts: `Agent.permission_mode` / create field; the
      create form's write checkbox becomes a mode <select> (manual|edit|auto,
      default manual); the detail "writes" row -> "mode".
- [x] Tests: migrate every `write_enabled` reference; add a legacy-migration test
      (persisted write_enabled=true -> permission_mode edit) and a per-backend
      flag-mapping test (manual->read-only/default, edit->workspace-write/
      acceptEdits, auto->danger-full-access/bypassPermissions).
- [x] Full suite + npm run ci green; close-out.

## Definition of Done

- An agent's write posture is `permission_mode` (manual|edit|auto, default
  manual); `write_enabled` is gone (test: `agent_store_permission_mode_default`).
- Each mode maps to the right per-backend flag
  (test: `codex_backend_permission_mode_flags`,
  `claude_backend_permission_mode_flags`).
- A legacy `write_enabled=true` record migrates to `edit`
  (test: `legacy_write_enabled_migrates_to_edit`).
- Full suite + `npm run ci` green.

## Close-out

What changed (replaced the write_enabled boolean with a permission mode across 10 files):
- config.py: `PERMISSION_MODES = (manual, edit, auto)` + `normalize_permission_mode`.
- backends.py: `_codex_sandbox_for` / `_claude_permission_mode_for` maps (values
  verified live via --help); `stream(..., permission_mode="manual")` replaces
  `write_enabled` on the protocol + all backends; ClaudeBackend ALWAYS passes
  `--permission-mode <mapped>`.
- agent_store.py: `AgentRecord.permission_mode: PermissionMode = "manual"`
  replaces `write_enabled`; create/update take `permission_mode` (normalized);
  MIGRATE legacy `write_enabled` -> edit/manual on load (before validate).
- app.py: `AgentCreate`/`AgentUpdate.permission_mode` (Literal) replace
  write_enabled; run passes `permission_mode=agent.permission_mode`.
- mcp_server.py: status shows `mode: <permission_mode>`.
- common.ts + agents-view.ts: `Agent.permission_mode`; create form's write
  CHECKBOX -> a mode <select> (manual|edit|auto, default manual); detail
  "writes" row -> "mode".
- Tests: migrated every write_enabled ref; added per-backend flag-mapping tests
  (manual->read-only/default, edit->workspace-write/acceptEdits,
  auto->danger-full-access/bypassPermissions), a mode-default/normalize test, and
  a legacy write_enabled->edit migration test.

Mapping (verified live: codex `read-only|workspace-write|danger-full-access`,
claude `default|acceptEdits|bypassPermissions`):
  manual -> read-only / default    (the default posture)
  edit   -> workspace-write / acceptEdits
  auto   -> danger-full-access / bypassPermissions

Design:
- Default is `manual` (read-only) - the operator's revised decision ("start in
  manual"), superseding the earlier "write default ON".
- Probed the exact flags LIVE before wiring (the promoted lesson): codex sandbox
  values and claude permission-mode values came from `--help`, not a guess.
- claude "default" in headless is a weaker read-only than codex's sandbox - noted
  in the code; a live check when running a real claude agent is warranted.

Result: 255 backend (+2) + 135 frontend tests, ruff + mypy clean.

Self-reflection: a wide but mechanical rename; the one real risk (a legacy
write_enabled agent silently becoming read-only because the field is gone) was
handled by migrating BEFORE model_validate, and pinned by a test.
