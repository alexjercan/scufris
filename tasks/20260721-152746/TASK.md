# Backend cleanup: drop the codex exec mode (app_server-only) + refresh .env.example & README for Agents v2

- STATUS: CLOSED
- PRIORITY: 37
- TAGS: agents,backend,docs
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

User feedback (2026-07-21): "I want to use only the app_server one and have it
named codex (same as claude is just claude - we already do that in the
frontend)." Drop the `exec` mode from the per-agent `CodexBackend` so it is
always `app_server`, surfaced as "codex". Then refresh `.env.example` and
`README.md` for the Agents v2 world (per-agent agents, claude backend,
permission modes, model-follows-backend, the chat).

## Investigation done

- Nothing constructs `CodexBackend("exec")` - `get_backend` always builds
  `CodexBackend()` (default app_server). So the `exec` MODE of CodexBackend is
  dead for the per-agent path and can be dropped now.
- BUT `exec` is NOT fully removable yet: the LANDING orchestrator `CodexCliAgent`
  (agent.py) still drives `codex exec` via `_run_codex_exec` / `_stream_codex_exec`
  for the non-streaming + streaming landing chat. Those stay until B5 migrates the
  orchestrator onto the backend interface. Answer to "do we use exec elsewhere":
  yes, the landing agent - so this task drops the CodexBackend MODE, not the codex
  exec runners.

## Steps

- [x] backends.py: remove `CodexMode`/the `mode` param + the `exec` branch from
      `CodexBackend` (always `_stream_app_server`). Keep `name = "codex"`. Leave
      `_stream_codex_exec`/`_run_codex_exec` in agent.py (still used by the
      landing `CodexCliAgent`) - add a comment that they are landing-only pending
      B5.
- [x] config.py: drop `"exec"` from `agent_backend`'s Literal (keep
      `app_server` for the landing agent + `mock`), or note why it stays. Confirm
      no other code reads a codex `exec` mode.
- [x] Doc-surface sweep + refresh: `.env.example` (remove `exec` as a backend
      option; add SCUFRIS_CLAUDE_MODEL / SCUFRIS_ENABLE_MOCK_BACKEND / permission
      modes; correct the codex-only framing) and `README.md` (the agent section:
      per-agent agents on `/agents`, codex + claude backends, permission modes,
      the chat, model-follows-backend). Grep every doc surface for `exec`.
- [x] Tests: CodexBackend has no exec mode (constructs app_server only); the
      backend/get_backend suites stay green.

## Definition of Done

- `CodexBackend` is app_server-only, named "codex"; no per-agent path selects
  exec (test: construct + get_backend; grep shows no `CodexMode`/`"exec"` in the
  per-agent path).
- `.env.example` and `README.md` reflect Agents v2 (cmd: `grep -n "claude\|permission\|app_server" .env.example README.md`; no stale `exec` backend option).
- Full check suite green.
- manual: README reads correctly for the current UX.

## Notes
- Depends on: none hard, but coordinate with B5 (which removes the landing
  CodexCliAgent's codex exec usage entirely).
- Relevant: scufris/backends.py (CodexBackend, CodexMode), scufris/config.py
  (agent_backend Literal), scufris/agent.py (_stream_codex_exec - landing only),
  .env.example, README.md.
- Close-out: dropped the exec MODE (not the exec RUNNER). `CodexBackend` lost its
  `mode` param + `CodexMode` (always app_server); `agent_backend` Literal is now
  `["app_server", "mock"]` (config + AgentConfigUpdate) with a
  `field_validator(mode="before")` coercing a legacy/persisted "exec" ->
  "app_server" so an existing override never fails to load; `build_agent` lost
  its exec-stream branch. The `_stream_codex_exec`/`_run_codex_exec` RUNNERS stay
  (agent.py) - the landing `CodexCliAgent`'s non-streaming `chat()` still uses
  `_run_codex_exec`, and they are directly tested; B5 will migrate the landing
  orchestrator off them. Tests: repointed the CodexBackend cwd/session +
  permission-mode tests to `_stream_app_server` (the backend's only runner now);
  swapped settings-store/app "exec" fixtures to "app_server"/"mock"; added a
  coercion test (`Settings.model_validate({"agent_backend":"exec"})` ->
  app_server). Docs: `.env.example` reframed to Agents (codex/claude, permission
  modes, claude_model, enable_mock_backend; exec noted as coerced) + README
  "Agents" section rewritten. Frontend already had no exec (F2/MB1). 271 backend
  tests green.
