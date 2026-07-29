# Gate the dashboard behind an authenticated session

- STATUS: OPEN
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

- [ ] Record the supported deployment boundary: loopback-only development, and
      authenticated LAN access as the deployed mode. Name the unsupported
      patterns (public exposure, shared host) explicitly.
- [ ] Choose the identity mechanism and record it in `DECISION.md`: single
      operator, credential source (sops-nix secret consistent with how
      `SCUFRIS_TELEGRAM_BOT_TOKEN` is already delivered), session cookie
      properties, expiry, and revocation.
- [ ] Require an authenticated session for every mutating and sensitive-read
      route, with a single enforcement point rather than per-route decoration.
- [ ] Keep loopback-only development explicit and low friction so `pytest`,
      examples, and the mock backend do not each need a login dance.
- [ ] Add CSRF protection, secure/`SameSite` cookie behavior, origin validation
      on state-changing requests, and login throttling.
- [ ] Keep the SSE/streaming endpoints and the Telegram bridge working under the
      session model (Telegram authenticates by chat id, not cookie).
- [ ] Add bypass tests: unauthenticated route sweep, session fixation, CSRF,
      cross-origin, and expired/revoked session.

## Definition of Done

- Every mutating and sensitive-read route rejects unauthenticated and
  CSRF-invalid requests, proven by an enumerate-the-app-routes sweep rather than
  a hand-written list (test: `test_authenticated_session_and_csrf_boundary`).
- Loopback-only development mode stays explicit and documented
  (test: `test_loopback_only_auth_policy`).
- Streaming and Telegram flows still work with authentication on
  (test: `test_authenticated_streaming_and_telegram_bridge`).
- manual: logging in from a phone on the LAN is bearable enough that the
  operator does not disable it.

## Notes

- Epic: 20260729-124655.
- Carved out of 20260729-102208 (which keeps secret references and redaction).
- Blocks every mutating child of this epic.
- `scufris/app.py` already has one HTTP middleware (oversized-upload rejection)
  - put enforcement there rather than sprinkling dependencies across 59 routes.
  Coordinate with 20260729-103712 if the router split lands first.

## Flow State

- FLOW STEP: PLANNING
