# Gate the dashboard behind an authenticated session

- STATUS: CLOSED
- PRIORITY: 70
- TAGS: feature,v0.2.0,security,auth,backend

## Story

As the operator, I want the dashboard to require an authenticated session, so
that giving Scufris power over the host does not hand that power to everything
on my LAN.

As deployed from `~/personal/nix.dotfiles`, Scufris binds `0.0.0.0:8000` and the
host firewall explicitly accepts `192.168.0.0/24 -> 8000`. There is no HTTP
authentication today: anything on the network can already create agents, run
turns, and drive the Telegram-connected orchestrator. The host operator epic
adds service restarts and `nixos-rebuild switch` on top of that surface, so this
task is its hard prerequisite, not a nice-to-have.

Scope is deliberately the AUTHENTICATION half only. The secret-reference store
and redaction work stays in 20260729-102208 for the plugin epic.

## Steps

- [x] Record the supported deployment boundary and the chosen identity mechanism
      in `DECISION.md`: loopback-only development vs authenticated LAN access,
      the unsupported patterns (public exposure, shared host), the credential
      source (the existing sops dotenv `secrets/scufris.env`), session cookie
      properties, expiry, and revocation.
- [x] Add the config surface: `auth_mode` (`auto`/`required`/`disabled`),
      `auth_password_hash`, and the session lifetime knobs, with fail-closed
      startup validation (a non-loopback bind without a credential refuses to
      start, and `disabled` is refused off loopback).
- [x] Add `scufris/auth.py`: scrypt hash/verify (stdlib `hashlib`, no new
      dependency), a revocable server-side session store under `state_dir`, CSRF
      token issue/verify, and failed-login throttling.
- [x] Add the `scufris hash-password` CLI subcommand so the operator generates
      the hash without the password ever reaching the repo or a log.
- [x] Add the auth routes: `POST /api/auth/login`, `POST /api/auth/logout`,
      `GET /api/auth/session`.
- [x] Require an authenticated session at a SINGLE enforcement point (one HTTP
      middleware, deny by default with a small public allowlist) rather than
      per-route decoration across 59 routes.
- [x] Keep the machine callers working: the MCP tool servers and the in-process
      operator tool console call this app's own HTTP API with no credential
      today (`scufris/mcp_common.py::_api_call`, base injected at
      `scufris/agent.py`). Inject a per-process `SCUFRIS_API_TOKEN` and accept it
      as a bearer token at the same enforcement point.
- [x] Keep loopback-only development explicit and low friction so `pytest`,
      examples, and the mock backend do not each need a login dance.
- [x] Add CSRF protection (double-submit cookie plus required header), secure/
      `SameSite` cookie behavior, origin validation on state-changing requests,
      and login throttling.
- [x] Keep the SSE/streaming endpoints and the Telegram bridge working under the
      session model (Telegram authenticates by chat id, not cookie).
- [x] Add the login page, the shared 401 -> redirect in the frontend API helper,
      and a logout control.
- [x] Add bypass tests: unauthenticated route sweep, session fixation, CSRF,
      cross-origin, expired/revoked session, and throttling.
- [x] Update the doc surfaces (`.env.example`, README, AGENTS.md deployment
      note) and add an `examples/` login round-trip script.
- [x] Wire the deployment in `~/personal/nix.dotfiles` (commit only - never push,
      never activate) and record the exact secret line the operator must add.

## Definition of Done

- Every mutating and sensitive-read route rejects unauthenticated and
  CSRF-invalid requests, proven by an enumerate-the-app-routes sweep rather than
  a hand-written list (test: `test_authenticated_session_and_csrf_boundary`).
- Loopback-only development mode stays explicit and documented
  (test: `test_loopback_only_auth_policy`).
- Streaming and Telegram flows still work with authentication on
  (test: `test_authenticated_streaming_and_telegram_bridge`).
- The MCP tool servers and the operator tool console still reach the API with
  authentication on (test: `test_mcp_tools_reach_the_api_under_auth`).
- A non-loopback bind with no credential configured refuses to start rather than
  serving open (test: `test_non_loopback_bind_without_credentials_refuses_to_start`).
- manual: logging in from a phone on the LAN is bearable enough that the
  operator does not disable it.

## Notes

- Epic: 20260729-124655.
- Carved out of 20260729-102208 (which keeps secret references and redaction).
- Blocks every mutating child of this epic.
- `scufris/app.py` already has one HTTP middleware - request logging, not the
  oversized-upload rejection this note originally claimed (verified at work
  time). Put enforcement in a second middleware rather than sprinkling
  dependencies across the 58 routes. Coordinate with 20260729-103712 if the
  router split lands first.
- Base suite on the pristine base commit (e817e8b): `mypy .` and
  `python -m pytest` GREEN; `ruff format --check .` RED on three untouched files
  (`scufris/enums.py`, `tests/test_mcp_server.py`, `tests/test_supervisor.py`).
  That red is inherited, not this task's.

## Flow State

- FLOW STEP: DONE
- PLAN STATUS: APPROVED
