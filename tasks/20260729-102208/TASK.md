# Add protected secret references and redaction

- PRIORITY: 0
- TAGS: feature, backlog, security, auth, plugins
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Story

As the owner of a personal assistant with access to private integrations, I
want authenticated browser sessions and protected secret references, so that
future remote access or plugins do not turn local automation into an
unauthenticated secret-exposure surface.

## Steps

- [ ] Record the supported deployment boundary: loopback-only mode,
      authenticated LAN/reverse-proxy mode, and explicitly unsupported exposure
      patterns.
- [ ] Choose and document local authentication, session, password/token
      storage, logout/revocation, and reverse-proxy identity behavior.
- [ ] Require authentication for mutating and sensitive read endpoints outside
      explicitly configured loopback-only development mode.
- [ ] Add CSRF protection, secure cookie behavior, origin validation, login
      throttling, and safe redirect handling.
- [ ] Add a secret-reference store/API that never returns values after write and
      supports missing/rotated/revoked states.
- [ ] Redact secrets from logs, events, presets, agent proposals, plugin health,
      errors, exports, and audit records.
- [ ] Add threat-model documentation and integration tests for common bypass,
      fixation, CSRF, path, and redaction failures.

## Definition of Done

- Protected routes reject unauthenticated and CSRF-invalid requests
  (test: `test_authenticated_session_and_csrf_boundary`).
- Secret values never appear in API responses, logs, activity events, exports,
  persisted presets, or agent proposals
  (test: `test_secret_values_are_redacted_everywhere`).
- Loopback-only development remains explicit and low-friction
  (test: `test_loopback_only_auth_policy`).
- The documented deployment model makes it unambiguous when Scufris
  may be safely exposed beyond localhost (manual: user check).

## Notes

- Epic: 20260729-102204.
- Depends on: 20260729-102147 and 20260729-102207.
- Record the identity and secret-storage boundary in `DECISION.md`.
- This task does not yet authorize individual plugin actions; that belongs to
  20260729-102919.
- SCOPE CHANGE (2026-07-29 backlog review): the browser-authentication half of
  this task was pulled forward into v0.1.0 as 20260729-125015, because the
  dashboard is deployed LAN-reachable and unauthenticated and the host operator
  epic adds mutating power on top of it. What remains here is the SECRET half:
  the secret-reference store, its lifecycle states, and redaction across logs,
  events, exports, presets, agent proposals, plugin health, and audit records.
  Steps 1 to 4
  and their proofs now belong to 20260729-125015; re-plan this task against the
  remainder when the plugin epic is scheduled.
