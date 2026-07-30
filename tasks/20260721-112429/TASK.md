# B1: backend surface cleanup (Codex/Claude only, mock dev-flag, drop exec, per-backend model, labels)

- STATUS: CLOSED
- PRIORITY: 50
- TAGS: agents,backend
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED


## Goal

Clean the backend model surface to the two user-facing backends:
- `get_backend("codex")` -> `CodexBackend("app_server")`; `"claude"` ->
  `ClaudeBackend`. DROP `exec` from the user surface. `mock` resolvable and
  listed ONLY when a dev flag is on (`SCUFRIS_ENABLE_MOCK_BACKEND`, default off).
- `KNOWN_BACKENDS` becomes {codex, claude} (+ mock when flagged). Back-compat:
  map legacy persisted `app_server`/`exec` -> `codex` on load.
- Per-backend DEFAULT MODEL: codex -> settings.agent_model; claude -> a claude
  default (fixes the "claude shows gpt-5.5" bug).
- Friendly LABELS in one map so no UI shows raw ids.

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (decision 2; recommendation "Backend model surface").
- Fixes bugs #1 (model) and #3 (raw backend names) at the data layer.

## Steps

- [x] config.py: `enable_mock_backend: bool = False` (dev flag) and
      `claude_model: str = ""` (claude default). Add light helpers
      `canonical_backend(name)` (app_server/exec/codex -> codex; claude; mock) and
      `available_backends(settings)` (["codex","claude"] + mock when flagged) -
      kept in config (no heavy imports) so agent_store need not import backends.
- [x] backends.py `get_backend(name)`: normalize via `canonical_backend`;
      codex -> CodexBackend (app_server runner), claude -> ClaudeBackend, mock ->
      MockBackend; unknown -> ValueError. `CodexBackend.name = "codex"` (friendly),
      exec dropped from the user surface (the exec runner stays in agent.py for
      the non-streaming path).
- [x] agent_store.py: validate create's backend against
      `available_backends(settings)` (canonicalized); normalize each record's
      backend on load (legacy app_server/exec -> codex); per-backend default model
      (`_default_model`: claude -> settings.claude_model, else settings.agent_model)
      - fixes the "claude shows gpt-5.5" bug.
- [x] common.ts: `AGENT_BACKENDS = ["codex","claude"]` (the two canonical user
      choices; friendly Capitalized labels + a BACKEND_LABELS display map land in
      F2). Update the create-form default assertions accordingly.
- [x] Tests: rewrite test_backends get_backend/CodexBackend around the new
      surface; agent_store backend validation + mock-flag + per-backend model;
      add `enable_mock_backend=True` to the mock test settings helpers
      (test_app/_mock_settings, test_mcp_server/_seed_agent, test_agent_store);
      the create-form test default -> "codex".
- [x] Full backend suite + `npm run ci` green; close-out.

## Definition of Done

- `get_backend("codex")` resolves to a CodexBackend, `get_backend("claude")` to
  ClaudeBackend; legacy `app_server`/`exec` normalize to codex; `exec`/unknown as
  a NEW create is rejected (test: `get_backend_normalizes_and_resolves`).
- `mock` is creatable only when `enable_mock_backend` is on
  (test: `mock_backend_gated_by_flag`).
- A claude agent gets a claude default model, not "gpt-5.5"
  (test: `claude_agent_default_model`).
- Full suite passes (cmd: `nix develop --command bash -c "ruff check . && mypy .
  && pytest -q"`) + `npm run ci` in web/.

## Close-out

What changed:
- config.py: `claude_model` (claude default) + `enable_mock_backend` flag, and
  light helpers `canonical_backend` (app_server/exec/codex -> codex; claude;
  mock), `available_backends(settings)`, `default_model_for(settings, backend)` -
  in config so agent_store validates/normalizes without importing the backend
  runners.
- backends.py: `CodexBackend.name` is now "codex" (friendly) with an internal
  app_server default mode; `get_backend` normalizes via `canonical_backend`
  (legacy codex modes -> codex), resolves codex/claude/mock, else ValueError.
- agent_store.py: create/update validate the canonicalized backend against
  `available_backends` (mock only when flagged); each record's backend is
  normalized on load; per-backend default model (fixes the claude "gpt-5.5" bug).
  Removed the stale `KNOWN_BACKENDS` set.
- common.ts: `AGENT_BACKENDS = ["codex","claude"]` (the two user choices).
- Tests: rewrote get_backend/CodexBackend around the surface; added mock-flag
  gating, per-backend model, legacy-normalize-on-load; added
  `enable_mock_backend=True` to the mock test settings helpers; create-form
  default -> "codex".

Design:
- `get_backend` RESOLVES mock always (a persisted mock agent must still run); the
  flag only gates CREATE (in the store) and the UI picker - so an existing mock
  agent keeps working if the flag is later turned off.
- Kept the codex `exec` runner in agent.py (non-streaming chat path) - only
  dropped it from the user-facing backend surface.
- The friendly display labels (Capitalized "Codex"/"Claude") + a BACKEND_LABELS
  map land with the cards in F2; B1 is the data/resolution layer.

Result: 253 backend (+3) + 135 frontend tests, ruff + mypy clean.

Self-reflection: putting the canonicalize/available helpers in config (not
backends) avoided an agent_store -> backends import that would have made the MCP
server's startup heavier - a small call that respected the A5 "keep MCP imports
light" lesson.
