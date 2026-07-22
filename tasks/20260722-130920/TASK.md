# account panel auth_mode is codex-centric for every agent - make it backend-aware

- STATUS: OPEN
- PRIORITY: 50
- TAGS: agents,backend,frontend,bug

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
