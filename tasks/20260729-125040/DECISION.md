# Decision: one host agent holds the mutating tools; three tasks build the approvals

- DATE: 20260730-104520
- STATUS: ACCEPTED
- TASK: 20260729-125040
- TAGS: decision, host, agents, telegram, frontend, v0.2.0

## Context

The epic's remaining agency work was seeded as ONE child task: "add the host
operator agent and its approval surfaces". Reading it against the code showed it
spans three separable surfaces (a new agent audience in `scufris/agent.py`, a web
page, and a Telegram interaction model), and that four of its steps could not be
built without settling a fork first - each fork having candidates that are
mutually exclusive, not interchangeable.

What is already shipped, and therefore NOT in question here:

- `POST /api/host/actions{,/{id}/approve,/deny,/cancel,/revert}` and
  `/api/host/audit` exist and are operator-only (`auth.OPERATOR_ONLY_PATTERN`).
  The web surface is a UI over routes that already work.
- `HostActionStore` (`scufris/host_actions.py`) is the app's in-memory request-side
  registry; the helper owns proposals, applies and the audit log.
- The MCP audience split is PHYSICAL (`agent.scufris_mcp_servers`): an
  orchestrator turn registers `scufris` + `den`, a sub-agent turn registers only
  `agent`. A tool reaches an audience by being on a server that audience
  registers, never by a runtime filter.
- `request_input` / `report_back` / `pending_agents` / `message_agent` already
  carry a blocked sub-agent's question to the orchestrator and the answer back.

## Decision

### 1. The task is re-cut into three child tasks

20260729-125040 keeps the id and becomes the agent plus the shared decision core;
two new children build the surfaces on top of it.

| task | priority | what it builds |
|------|----------|----------------|
| 20260729-125040 | 45 | the host agent, orchestrator delegation, and the ONE approval service both surfaces call (confirmation rules, requester notification, queue edges, restart recovery) |
| 20260730-104520 | 44 | the dashboard approval queue and audit/rollback surface |
| 20260730-104524 | 43 | host approvals over Telegram |

Why split: three surfaces in one review round means the reviewer holds a new agent
audience, a new web page and a new bot interaction model at once - and 125029 (the
framework, one surface) already cost 3 rounds and 25 findings. The shared
semantics do NOT fragment, because they all land in the first task: the surfaces
render a confirmation requirement the core computes rather than each deciding what
"stronger acknowledgement" means.

The cost accepted: `test_host_approval_from_either_surface` cannot pass until the
third task lands, so it belongs to that task and the epic's Done Means is met at
the end of the three, not the first.

### 2. The mutating host tools move to the host agent. The orchestrator keeps read-only.

The propose-side tools - `propose_host_action`, `propose_nixos_change`,
`host_action_status`, `nixos_change_status`, `host_action_audit` - move OFF the
orchestrator's `scufris` server onto a new `host` MCP server registered only on a
host-agent turn. The read-only inspection tools (`host_stats`, `disk_usage`,
`list_processes`, `host_units`, `host_failed_units`, `host_unit_status`,
`host_journal`, `host_storage`, `host_largest_directories`,
`host_reclaimable_space`, `host_network`, `host_thermal`, `host_what_provides`,
`host_generation_diff`, `host_flake_status`) stay on the orchestrator AND are also
registered for the host agent.

Because the split is physical, a tool cannot be on both audiences without being
registered twice - so this is a real removal from the orchestrator, not a filter.

Consequences, stated rather than discovered later:

- "why is this box hot" and "what filled the disk" stay a DIRECT orchestrator
  answer (20260729-125024's manual acceptance is unaffected).
- "restart that service" from chat now costs a delegation round-trip: the
  orchestrator spawns/messages the host agent, which proposes. That is slower
  than proposing inline, and it is the accepted price for the propose/preview/
  approve contract living in exactly ONE steering preamble and one audience.
- Rejected: keeping the tools on both audiences. Two audiences means the contract
  is stated twice and can drift in one place, and the host agent then adds only a
  separate context rather than being the mutating path.
- Rejected: moving inspection too. It would make every question about the machine
  a sub-agent spawn and regress the shipped fast path.

### 3. On Telegram, the allowlisted chat IS the operator

Approve/deny from an allowlisted chat is an operator decision. The bot token and
the chat allowlist are both sops secrets in the same dotenv as the password hash,
and the audit records `operator:telegram:<chat_id>` so the record says which
surface decided.

This is a REAL privilege increase and is accepted deliberately: today an
allowlisted chat can drive the orchestrator (so it can propose) but cannot
approve; after this it can approve a root action with no password. The threat it
accepts is a compromised Telegram account; the threat model already notes that
these controls do not defend against a compromised operator account (the `docker`
group is root-equivalent on this machine).

- Rejected: `/unlock <password>` minting a short-lived Telegram approval session.
  It puts the operator password into a chat log to buy protection against an
  attacker who already owns the operator's phone or Telegram account.
- Rejected: notify-only with the decision on the web. It contradicts the step's
  own requirement and makes the phone path useless exactly when the operator is
  away from the desk.

What "no privileged shortcut for being on Telegram" therefore means, concretely:
Telegram does NOT get a second decision path. Both surfaces call one
`HostApprovalService`; the web route supplies a session-derived actor and Telegram
supplies a chat-derived one, and every rule after that point - already decided,
expired, drifted, one-way acknowledgement, race - is the same code.

### 4. Restart recovery reads the helper: a new read-only `list_pending` verb

`HostActionStore` stays in memory, and the app rebuilds it from the helper at
startup and on demand through a new read-only hostd verb that lists the proposals
the helper still holds PENDING.

Why this and not persistence: "the helper holds every proposal" is the invariant
that makes "preview one thing, apply another" unreachable. Writing the queue to an
app-owned state file would create a second, app-writable record of what was
proposed and decided, sitting next to a root-owned audit log that exists precisely
because the app may be the thing that misbehaved.

The new verb builds no argv, takes no proposal id, and returns the same
`ProposalView` frames the helper already emits - it is a read of state the helper
already keeps, which is why it is acceptable on the root socket. It needs its
authenticated-secret check and its refusal-coalescing like every other verb.

- Rejected: persisting `HostActionStore` (second source of truth, contradicts
  20260729-125029's recorded reasoning).
- Rejected: narrowing the DoD to "pending proposals are not durable". A ten-minute
  TTL means a restart during a deploy silently strands a real approval, and the
  epic's promise is a visible pending-decision queue.

### 5. A pending approval is a pending agent, routed to the OPERATOR

The host agent proposes and ends its turn; the app records the outcome as BLOCKED
with the rendered proposal as its message. AMENDED DURING THE BUILD: this section
originally said "WAITING, marked as awaiting the operator", and the marker turned
out to already exist - `AgentState.BLOCKED` was defined and unused, documented as
"waiting on an approval". So the distinction needs no new field: WAITING means the
ORCHESTRATOR owes an answer, BLOCKED means a human does. That marking is
load-bearing in two directions:

- the wake bridge must NOT wake the orchestrator to answer it, and
  `pending_agents` must show it as "waiting on the operator, you cannot decide
  this" - otherwise the orchestrator answers "yes, approved, go ahead", which it
  has no authority to say (only a session/chat operator can reach `apply`).
- the decision, when it comes, resumes the host agent's session with the outcome
  or the denial reason, through the same `_launch_agent_turn` path
  `message_agent` uses, deferred if a run is already active (the `WakeBridge`
  deferral shape).

So the answer to "reuse the existing machinery or add a parallel path" is: reuse
the shape and the transport, add the AUDIENCE field. A pending approval is the
same shape as a pending question addressed to a different decider.

### 6. The confirmation requirement is computed once, in the core

The core exposes, per action record, what confirming it requires:

- ordinary confirmation, with the undo sentence shown as written, for anything
  that either can be undone or is mere service control;
- the strong path - the approve call must carry an explicit acknowledgement token,
  and a call without it is refused with 422 by the SERVICE - for an action that
  DESTROYS something: irreversible and not service control. Today that is R2,
  disposable cleanup.

CORRECTED DURING THE BUILD. This section first said the rule keys on
`reversal.possible` alone, "so a future irreversible verb inherits the strong path
automatically", and explicitly rejected using the risk class. Running it against
the real helper refuted the premise: for R1, `possible=False` is the NORMAL answer
rather than the alarming one - restarting or reloading an active unit ends where it
started, so there is no state to restore, and starting an already-active unit
reports no undo because it changed nothing (`hostd/preview.py`). The pure rule
therefore demanded a typed acknowledgement for every service restart, which is
wrong about the risk and self-defeating: a warning that fires on the routine act is
the reason nobody reads the one on `gc_store`.

So R1 is carved out, and the carve-out is stated in the code with the measurement
that forced it. What is NOT given up: the surfaces still SHOW whether an action can
be undone (`Confirmation.no_undo` and `undo` are separate from the style), so the
honesty is unchanged and only the friction is proportionate.

`POST /api/host/actions/{id}/approve` therefore grows an optional body; existing
callers of the reversible path are unaffected, and `examples/host_action.py` plus
the API tests are updated in the same task.

## Consequences

- `scufris/agent.py`'s audience concept becomes three-valued (orchestrator, host,
  agent) instead of a boolean plus an id, and `sessions.py` gains a third
  steering preamble as ONE `[scufris-tools]` block (the ledger's
  `orchestrator-steering-is-one-block-two-clauses` lesson).
- The orchestrator's steering loses its propose clause and gains a delegate-to-
  the-host-agent clause naming the real tool signatures
  (`ground-steering-text-in-the-real-tool-signatures`).
- `nix build .#hostd-vm-test` grows a `list_pending` assertion: a verb that reads
  the helper's real state on a real socket is exactly what the VM test is for.
- AGENTS.md's privileged-actions section gains the audience split, the Telegram
  credential and the new verb; the epic's Done Means 2 and 4 are unchanged.
