# Add the host action framework with preview approval and audit

- STATUS: OPEN
- PRIORITY: 55
- TAGS: feature,v0.2.0,host,security,backend

## Story

As the operator, I want every host change to arrive as a proposal I can preview
and approve, so that an agent with access to my machine is a colleague asking
before it acts rather than a process with root and good intentions.

This is the mechanism the rest of the epic runs on:

    propose -> preview -> approve -> apply -> audit -> roll back

It is deliberately built for ONE consumer (host actions) rather than as a
general capability system. Generalizing waits for a second consumer.

## Steps

- [ ] Implement the typed host action: identity, risk class, arguments,
      requester (agent and run), preview payload, and reversal descriptor,
      following the taxonomy decided in the host spike.
- [ ] Implement preview generation per risk class, with a defined result when a
      class cannot honestly preview (that absence must be visible to the
      approver, not silently empty).
- [ ] Implement the approval gate at the EXECUTION boundary: an unapproved
      action cannot execute even if a prompt, a tool description, or a UI state
      says otherwise.
- [ ] Implement execution with the privilege mechanism chosen in the spike:
      timeouts, cancellation through the existing supervisor, streamed output,
      and a structured result.
- [ ] Implement the append-only audit: requested, denied, approved, applied, and
      failed events with actor, agent, run, arguments, result, duration, and
      reversal reference. Redact anything secret-shaped.
- [ ] Implement reversal: each applied action records how to undo it (or records
      that it cannot be undone), and the undo path is exercised by tests.
- [ ] Define approval expiry and scope: a stale proposal cannot be approved
      after the system moved underneath it, and approving one action does not
      approve the next.
- [ ] Add adversarial tests: forged action id, replay of an approved action,
      approval after system drift, concurrent approvals of the same proposal,
      cancellation mid-apply, and secret redaction in the audit.

## Definition of Done

- An action with no effective approval never reaches execution, regardless of
  the path it was requested through
  (test: `test_host_action_requires_preview_and_approval`).
- Approving a stale proposal is refused, and approvals do not replay
  (test: `test_host_action_approval_is_scoped_and_single_use`).
- Every requested, denied, approved, applied, and failed action produces a
  durable redacted audit record (test: `test_host_actions_are_audited`).
- Cancelling mid-apply leaves a recorded, consistent outcome rather than an
  unknown state (test: `test_host_action_cancellation_is_recorded`).
- manual: an approval prompt states plainly what will change and how it can be
  undone.

## Notes

- Epic: 20260729-124655.
- Depends on: the host spike (taxonomy, privilege, preview, rollback) and the
  dashboard authentication task.
- Persist through the transactional store from 20260729-102147; concurrent
  agents means concurrent proposals.
- Reuse `scufris/supervisor.py` for run lifecycle and cancellation instead of
  inventing a second execution path.
- The general capability-grant system is 20260729-102919 and stays in the
  backlog. Build this one concretely; generalize when there is a second consumer
  to generalize from.

## Flow State

- FLOW STEP: PLANNING
