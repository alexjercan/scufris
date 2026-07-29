# Add local authentication and protected secret references

- STATUS: OPEN
- PRIORITY: 0
- TAGS: feature,backlog,security,auth,plugins

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
- [ ] Redact secrets from logs, events, blueprints, plugin health, errors,
      exports, and audit records.
- [ ] Add threat-model documentation and integration tests for common bypass,
      fixation, CSRF, path, and redaction failures.

## Definition of Done

- Protected routes reject unauthenticated and CSRF-invalid requests
  (test: `test_authenticated_session_and_csrf_boundary`).
- Secret values never appear in API responses, logs, activity events, exports,
  or persisted blueprints (test: `test_secret_values_are_redacted_everywhere`).
- Loopback-only development remains explicit and low-friction
  (test: `test_loopback_only_auth_policy`).
- manual: the documented deployment model makes it unambiguous when Scufris
  may be safely exposed beyond localhost.

## Notes

- Epic: 20260729-102204.
- Depends on: 20260729-102205 and 20260729-102147.
- Record the identity and secret-storage boundary in `DECISION.md`.
- This task does not yet authorize individual plugin actions; that belongs to
  20260729-102919.

## Flow State

- FLOW STEP: PLANNING
