# Add capability grants approvals and action audit

- STATUS: OPEN
- PRIORITY: 0
- TAGS: feature,backlog,plugins,security,agents

## Story

As an operator, I want capabilities granted explicitly and consequential
actions approved and audited, so that an agent cannot silently gain access to
email, calendar, files, network services, or outward side effects.

## Steps

- [ ] Define typed capability identifiers and resource scopes for filesystem,
      network domains, secrets, email, calendar, process execution, and plugin
      tools.
- [ ] Compute an effective grant from plugin declarations, blueprint requests,
      project policy, agent role, and user overrides; default to deny.
- [ ] Define approval modes for read, draft, write, send/publish, destructive,
      and repeated actions, including scoped one-time and remembered grants.
- [ ] Enforce grants and approvals at the tool-execution boundary rather than
      relying on prompts or UI visibility.
- [ ] Persist an append-only audit event for requests, decisions, invocations,
      redacted arguments, results, actor, agent/run, and artifact references.
- [ ] Add approval queue APIs and UI with clear capability, target, reason,
      argument diff, expiration, deny, and revoke controls.
- [ ] Add confused-deputy, forged-tool, stale-grant, replay, secret-redaction,
      cancellation, and concurrent-approval tests.

## Definition of Done

- Undeclared, ungranted, or unapproved privileged actions never reach a plugin
  (test: `test_plugin_action_requires_effective_grant_and_approval`).
- Send/publish/destructive actions require explicit confirmation by default
  (test: `test_outward_actions_default_to_explicit_confirmation`).
- Every allowed or denied action has a durable redacted audit record tied to
  its agent and run (test: `test_capability_decisions_are_audited`).
- Revocation takes effect before the next invocation
  (test: `test_capability_revocation_is_immediate`).
- manual: the approval surface makes the action and its consequences clear.

## Notes

- Epic: 20260729-102204.
- Depends on: 20260729-102207, 20260729-102208, and 20260729-102203.
- Record the policy-composition and approval-mode choice in `DECISION.md`.

## Flow State

- FLOW STEP: PLANNING
