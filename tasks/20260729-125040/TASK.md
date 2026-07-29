# Add the host operator agent and its approval surfaces

- STATUS: OPEN
- PRIORITY: 45
- TAGS: feature,v0.2.0,host,agents,frontend

## Story

As the operator, I want a host agent I can talk to and an approval surface I can
reach from my phone, so that acting on the machine is one conversation with a
visible pending-decision queue, not a hunt through chats.

Every agent today is bound to a project working tree; the orchestrator alone
runs unbound, in the server's own directory. The host agent is a third thing: it
is bound to the MACHINE, carries the host toolset, and holds no project.

## Steps

- [ ] Add the host agent: host-scoped rather than project-scoped, with the host
      inspection and action tools and a system prompt that states the propose/
      preview/approve contract as its normal way of working.
- [ ] Let the orchestrator delegate to it and read its results through the
      existing `request_input` / `report_back` / `pending_agents` machinery,
      rather than adding a parallel communication path.
- [ ] Add the approval queue in the dashboard: pending proposals with their
      risk class, preview, requester, expiry, and approve/deny controls, plus
      the audit history with its rollback controls.
- [ ] Add approvals over Telegram: a pending proposal notifies, the preview is
      readable on a phone, and approve/deny works from the chat with the same
      enforcement as the web path (no privileged shortcut for being on Telegram).
- [ ] Make the risk class legible: a service restart and a system switch must
      not look identical in either surface.
- [ ] Handle the queue's edges: expired proposals, proposals whose agent run was
      cancelled, approval races between the two surfaces, and denial with a
      reason that reaches the requesting agent.
- [ ] Cover both surfaces at desktop and phone widths.

## Definition of Done

- A host agent proposes, the operator approves from either surface, and the
  action applies exactly once (test: `test_host_approval_from_either_surface`).
- Telegram approval enforces the same gate as the web path
  (test: `test_telegram_approval_uses_the_same_enforcement`).
- The queue survives a restart with proposals, expiries, and audit intact
  (test: `test_approval_queue_survives_restart`).
- A denial reaches the requesting agent with its reason so the agent can adapt
  instead of retrying blindly (test: `test_denial_reaches_the_requesting_agent`).
- manual: approving a real host change from a phone is clear enough to do
  confidently while away from the desk.

## Notes

- Epic: 20260729-124655.
- Depends on: the host action framework, the NixOS configuration flow, and the
  dashboard authentication task.
- Reuse the existing bidirectional agent machinery (`scufris/wake.py`,
  `pending_agents`, `report_back`) - a pending APPROVAL is the same shape as a
  pending question.
- Telegram already streams orchestrator turns with per-phase messages; approvals
  should feel like part of that conversation, not a second bot.

## Flow State

- FLOW STEP: PLANNING
