# Add the host operator agent and the approval decision core

- STATUS: CLOSED
- PRIORITY: 45
- TAGS: feature,v0.2.0,host,agents
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

As the operator, I want a host agent I can talk to and ONE place where a pending
decision is enforced, so that acting on the machine is a conversation with a
visible pending-decision queue, and so the two approval surfaces built on top of
this cannot drift into two different sets of rules.

Every agent today is bound to a project working tree; the orchestrator alone runs
unbound, in the server's own directory. The host agent is a third thing: it is
bound to the MACHINE, carries the host toolset, and holds no project.

This task was re-cut from "add the host operator agent and its approval surfaces":
the web and Telegram surfaces are now 20260730-104520 and 20260730-104524, and
what stays here is the agent, the delegation, and the decision core they both
call. `DECISION.md` in this folder records the four forks that re-cut it.

## Steps

- [x] Add the `host` MCP server carrying the host toolset: the read-only
      inspection tools plus the mutating `propose_host_action`,
      `propose_nixos_change`, their status readers and `host_action_audit` - and
      REMOVE those mutating tools from the orchestrator's `scufris` server. The
      audience split is physical, so this is a real move, not a filter.
- [x] Make the audience three-valued in `agent.scufris_mcp_servers`,
      `_mcp_overrides` and `_steer` (orchestrator / host / agent) instead of a
      boolean plus an id, and add the host steering preamble as ONE
      `[scufris-tools]` block stating the propose -> preview -> approve contract
      as its normal way of working. Name every tool with its real signature.
- [x] Add the reserved host agent (`HOST_AGENT_ID = "host"`): synthesized from
      settings like the orchestrator, no project binding (server cwd), refused as
      a created id, immune to update/delete, reachable at `/agents/host`.
- [x] Let the orchestrator delegate to it and read its results through the
      existing `run_agent` / `message_agent` / `request_input` / `report_back` /
      `pending_agents` machinery, and re-point the orchestrator's steering from
      "propose the action yourself" to "delegate it to the host agent".
- [x] Add the one approval service both surfaces call: approve, deny, cancel and
      revert, with the per-action confirmation requirement computed from
      `reversal.possible` (not from the risk letter), and a one-way approve
      refused unless it carries its acknowledgement token. Move the HTTP routes
      onto it so the web path has no rule of its own.
- [x] Route a pending approval to the OPERATOR: the requesting agent's outcome is
      WAITING marked operator-bound, the wake bridge does not wake the
      orchestrator for it, and `pending_agents` shows it as a decision the
      orchestrator cannot make.
- [x] Deliver the decision back to the requesting agent: resume its session with
      the applied result or with the denial and its reason, deferred while a run
      is active, so a denied agent adapts instead of retrying blindly.
- [x] Handle the queue's edges in the core: an expired proposal, a proposal whose
      agent run was cancelled, an approval race between two surfaces (one
      execution, one refusal), and drift since the preview.
- [x] Add the read-only `list_pending` verb to `scufris-hostd` and rebuild
      `HostActionStore` from it at startup and on demand, so a restart within a
      proposal's TTL does not strand an approvable action. Assert it in
      `nix build .#hostd-vm-test`.
- [x] Update the docs surfaces in THIS task: AGENTS.md (the audience split, the
      BLOCKED routing, the one decision path, the confirmation rule and the new
      verb), README (the host agent, who may propose, the blocked round trip),
      CHANGELOG, `examples/host_action.py` (now prints the confirmation
      requirement) and a new `examples/host_agent.py` that drives the whole round
      trip against the real app and a real helper - propose, the orchestrator
      being refused, the decision, and the turn the agent is resumed with.

## Definition of Done

- The host audience holds the mutating tools and the orchestrator does not,
  physically rather than by a filter
  (test: `test_host_audience_holds_the_mutating_tools`).
- The orchestrator delegates to the host agent and reads its result through the
  existing machinery, adding no parallel communication path
  (test: `test_orchestrator_delegates_to_the_host_agent`).
- A pending approval is never answerable by the orchestrator
  (test: `test_pending_approval_is_operator_bound`).
- A denial reaches the requesting agent with its reason so the agent can adapt
  instead of retrying blindly (test: `test_denial_reaches_the_requesting_agent`).
- A one-way action cannot be approved through the ordinary reversible-action
  confirmation (test: `test_one_way_action_requires_stronger_confirmation`).
- The queue survives a restart with proposals, expiries and audit intact
  (test: `test_approval_queue_survives_restart`).
- Two approvals racing on one proposal produce one execution and one refusal
  (test: `test_approval_race_yields_one_execution`).
- cmd: `python -m pytest`
- cmd: `nix flake check`
- cmd: `nix build .#hostd-vm-test` (needs KVM; the release pipeline is where it
  runs unattended)
- manual: asking the orchestrator to restart a service or change the config
  reaches the host agent, and the pending decision is visible and honest about
  what it would do.

## Notes

- Epic: 20260729-124655.
- Depends on: the host action framework (20260729-125029), the NixOS
  configuration flow (20260729-125035), and the dashboard authentication task
  (20260729-125015) - all landed.
- Blocks: 20260730-104520 (dashboard surface) and 20260730-104524 (Telegram),
  which render the confirmation requirement this task computes.
- `DECISION.md` (this folder): the tool audience, the Telegram credential, the
  restart-recovery mechanism, the pending-approval routing and the re-cut.
- Reuse the existing bidirectional agent machinery (`scufris/wake.py`,
  `pending_agents`, `report_back`) - a pending APPROVAL is the same shape as a
  pending question addressed to a different decider.
- Inherited from 20260729-125029 review round 2, R2.5: the framework carries
  `risk` and `reversal.possible`; the differentiated confirmation is computed
  here and rendered by the two surface tasks.
- Ledger lessons that bite here: keep each steering preamble ONE
  `[scufris-tools]` block (`orchestrator-steering-is-one-block-two-clauses`),
  ground steering text in the real tool signatures
  (`ground-steering-text-in-the-real-tool-signatures`), and a changed protocol
  signature reds every test double, not just the real implementations
  (`protocol-signature-change-hits-the-doubles`) - the audience change and the
  new verb both touch shared signatures.
