# account panel auth_mode is codex-centric for every agent - make it backend-aware

- STATUS: IN_PROGRESS
- PRIORITY: 50
- TAGS: agents, backend, frontend, bug

## Goal

Make the account panel's `auth_mode` backend-aware, the same way the health check
was made backend-aware (20260722-104034). Today `GET /api/agents/{id}/account`
returns `auth_mode=settings.agent_auth_mode` (the GLOBAL, codex/ChatGPT-centric
`chatgpt|api_key`) for EVERY agent, so a claude agent's settings page shows a
codex-flavored auth mode that does not describe how it actually authenticates.

## Why

Surfaced as a deferred boundary in the backend-aware-health task
(20260722-104034, see its RETRO/REVIEW): "the account panel's auth_mode stays the
global setting - the app does not model claude auth; deferred as a follow-up." The
health check now dispatches by the agent's backend; the account panel's auth_mode
is the last codex-assumption left on the unified settings page.

## Notes / scope to pin

- `AuthMode` (enums.py) currently models only codex/ChatGPT auth: CHATGPT,
  API_KEY. Claude Code has its own auth (subscription login vs an ANTHROPIC_API_KEY
  / a different CLI login). Decide whether to:
  (a) extend the model so each backend reports its own auth mode (probe/derive it
      per backend, like health does), or
  (b) make `auth_mode` optional/None for a non-codex agent (honest "not modeled"
      rather than a wrong codex value), mirroring how usage/memory are None for a
      non-codex agent in U2.
- `AccountInfo.model` is already per-agent and correct; only `auth_mode` (and its
  frontend render in the account panel) is in scope.
- A /spike may be warranted to map claude's actual auth surface (does scufris have
  any signal for it? does the claude backend expose login state?) before choosing
  (a) vs (b). Prefer the honest, testable option; do not invent a claude auth mode
  the app cannot actually observe.
- Cross-refs: 20260722-104034 (backend-aware health, the pattern to mirror),
  U2 20260721-234609 (the per-agent usage/memory/account endpoints + _agent_is_codex
  dispatch), the account panel render in web/src/agent-settings-view.ts.

## Direction (pinned with the user 2026-07-22)

The user authenticates via BROWSER/SUBSCRIPTION login: ChatGPT for codex,
claude.ai for claude. API keys are not used ("I don't have any") but should be
selectable "just in case". So the auth mode is per-backend and defaults to that
backend's subscription login.

## Spike findings (mapped 2026-07-22, folded in - no separate SPIKE task)

- `agent_auth_mode` (config.py:124, default CHATGPT) is CODEX-SPECIFIC: its
  docstring says "Only affects `scufris login`; codex holds the auth", and the one
  behavioral use (agent.py:462) is the codex `login --with-api-key` path keyed on
  SCUFRIS_OPENAI_API_KEY. Everywhere else it is DISPLAY-ONLY.
- Four sites report it, all with the GLOBAL value regardless of backend:
  `/api/agents/{id}/account` (app.py ~1272, THE per-agent bug), `/api/agent/account`
  (~1451), `/api/agent/info` (~607), `_agent_config` -> AgentConfig (~624).
- `AuthMode` (enums.py) models only CHATGPT|API_KEY. `ClaudeBackend` exists
  (backends.py:410, name "claude"); there is NO claude login flow / ANTHROPIC key
  wiring in the app - claude auth is held by the claude CLI, exactly as codex auth
  is held by codex.
- Frontend reads `auth_mode` in 3 interfaces (common.ts) + 2 renders: the account
  panel (agent-settings-view.ts:216, takes null fine) and the global config row
  (settings-view.ts:148, `configRow(label, value: string)` - needs null-handling).

## Decision

Model a claude subscription auth mode and dispatch the reported auth by the agent's
backend. This is the honest option: it reports the CONFIGURED/effective mode per
backend (the same status codex's mode has - a declared value the CLI enforces),
not an invented observation. Mock (no login) reports None, like usage/memory do.

BOUNDARY (out of scope, noted): no claude login flow / ANTHROPIC_API_KEY wiring is
added - `agent_claude_auth_mode` is the reported/effective mode only, mirroring
that `agent_auth_mode` "only affects `scufris login`". A real `scufris login` for
claude is a separate feature.

## Steps (/plan)

- [ ] `enums.py`: add `AuthMode.CLAUDE_AI = "claude_ai"` (claude.ai subscription
      login); refresh the enum docstring (it is no longer codex-only).
- [ ] `config.py`: add `agent_claude_auth_mode: AuthMode = AuthMode.CLAUDE_AI`
      (claude.ai vs api_key; reported/effective, the claude CLI holds real auth) +
      a pure helper `auth_mode_for_backend(settings, backend) -> AuthMode | None`
      (canonical codex -> `agent_auth_mode`; claude -> `agent_claude_auth_mode`;
      else/mock -> None). Refresh the `agent_auth_mode` comment to say it is the
      CODEX auth mode.
- [ ] `app.py`: dispatch every auth_mode site through the helper - the per-agent
      account by `agent.backend`; the global account/info/config by
      `settings.agent_backend`. Make `AccountInfo.auth_mode`, `AgentInfo.auth_mode`,
      `AgentConfig.auth_mode` -> `AuthMode | None`.
- [ ] Frontend `common.ts`: `auth_mode: string | null` on the 3 interfaces; add an
      `authLabel(mode)` helper (chatgpt -> "ChatGPT", claude_ai -> "claude.ai",
      api_key -> "API key", null/"" -> "-"). Use it in the account panel and the
      config row (configRow gets a `value ?? "-"` / label at the call site).
- [ ] `.env.example`: add `SCUFRIS_AGENT_CLAUDE_AUTH_MODE` next to the codex one if
      the codex var is documented there.
- [ ] Tests: `auth_mode_for_backend` per backend + overrides (config/unit);
      `GET /api/agents/{id}/account` returns claude_ai for a claude agent, chatgpt
      for a codex agent, None for a mock agent, and api_key when the backend's mode
      is set to api_key (app); the frontend `authLabel` mapping + the account panel
      showing "claude.ai" for a claude agent. Full suite (ruff/mypy/pytest, web
      `npm run ci`) green.

## Definition of Done

- A claude agent's account panel shows its claude auth (default "claude.ai"), a
  codex agent shows its codex auth (default "ChatGPT"), a mock agent shows "-"
  (not modeled); each is selectable to "API key". (test: the account endpoint per
  backend + the authLabel render; manual: a claude agent's settings shows claude.ai,
  not chatgpt.)
- `auth_mode` is dispatched by the agent's effective backend at every reporting
  site (per-agent account + the orchestrator's account/info/config), not the flat
  global value. (test: the config helper + the app endpoints.)
- No claude login flow is claimed that does not exist (the boundary above holds).
- Full backend + web suites green.
