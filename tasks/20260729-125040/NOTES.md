# Notes: the host agent and the approval decision core

What shipped, why it is shaped this way, and the two things the build corrected in
the plan. The forks the operator settled before any code existed are in
`DECISION.md`; this is the record of building it.

## What shipped

**A third audience.** `enums.Audience` + `audience_for(is_orchestrator, agent_id)`
is now the single place that answers "what kind of turn is this", and the four
dispatch sites read it instead of repeating the comparison:
`agent.scufris_mcp_servers` (which servers a turn registers), `agent._steer` (which
preamble rides the prompt), `mcp_health.servers_for_audience` (what the in-process
probe covers) and `app._mcp_servers_for_audience` (what the tool listings show).
The BACKEND signatures were left alone - `stream(..., is_orchestrator=, agent_id=)`
already carries everything the audience is derived from, so making the protocol
three-valued would have churned every backend and every test double
(`protocol-signature-change-hits-the-doubles`) to express what one function can.

**One definition of the host toolset.** `mcp_host_tools.py` holds the tool
functions and a registrar; `mcp_server` registers the INSPECTION half,
`host_mcp_server` registers inspection plus ACTIONS. Registration-by-call rather
than `@mcp.tool()`-at-definition is what lets the same function object serve one
audience or two without a runtime filter.

**One decision path.** `host_approvals.HostApprovalService` owns approve, deny,
cancel and revert; the HTTP routes are translators (derive the actor from the
credential, map the service's refusals onto statuses). `confirmation_for` computes
what confirming an action requires, so both surfaces render a requirement rather
than each inventing one. `decision_message` renders what the requesting agent is
told.

**The round trip.** A proposal marks the requesting agent `BLOCKED`; the decision
resumes it with the applied result or the denial reason, deferred if a turn is in
flight (drained where the wake bridge drains, past the serialize-key release). The
orchestrator sees the BLOCKED row in `pending_agents` and is refused if it tries to
answer it or acknowledge it.

**Restart recovery.** A read-only `list_pending` verb on the helper; the app
rebuilds its queue at startup and reconciles while listing (throttled by
`host_queue_refresh_seconds`). Additions only - an absence cannot be told apart
from expired / denied elsewhere / just applied.

## The two corrections the build forced

**1. The strong confirmation cannot key on `reversal.possible` alone.** The plan
said it should, explicitly rejecting the risk letter so "a future irreversible verb
inherits the strong path automatically". Then five existing tests went red at once
with a message that was the actual finding: a `unit_restart` of a RUNNING unit
reports `possible=False`, because the unit ends where it started and the process it
was running is gone. Reading `hostd/preview.py` showed three more R1 cases that
report no undo, two of which are no-ops ("starting it changes nothing to undo").

The pure rule therefore demanded a typed acknowledgement for every service restart.
That is wrong about the risk and self-defeating - the value of the warning on
`gc_store` is that it does not fire on the routine act. The rule is now "irreversible
AND not service control", the carve-out is documented in the code with the
measurement that forced it, and `Confirmation.no_undo` was split out from the style
so a surface still SAYS "no undo" for a restart without gating it. Honesty and
friction are now separate knobs, which is what the original conflated.

**2. The pending state already existed.** The plan said "a WAITING outcome marked
as awaiting the operator", implying a new field. `AgentState.BLOCKED` was already
defined and unused, with the comment "waiting on an approval". So WAITING means the
orchestrator owes an answer and BLOCKED means a human does, and the routing needed
no new field at all - only the three places that read the set of pending states.

## Difficulties, and what they cost

- **The tool split is a MOVE, and the tests import the moved names.** Extracting
  ~550 lines out of `mcp_server.py` broke `test_mcp_server.py` at import time. The
  cheap fix (re-export from the old module) would have left two names for one tool,
  so the tests moved with the code into `test_host_mcp_server.py`, including the
  monkeypatch targets (`scufris.mcp_server._inspector` ->
  `scufris.mcp_host_tools._inspector`). The extraction itself was done with a
  script slicing the exact line range rather than by hand, so no tool body was
  retyped.
- **A test that sends both a cookie and a bearer token is not testing an agent.**
  `test_pending_approval_is_operator_bound` first asserted the orchestrator gets a
  409 and got a 200: `TestClient` had the operator's session cookie from `_login`,
  and `_caller_is_agent` correctly answered "this is the operator with an
  Authorization header". A second client with its own cookie jar is what a real MCP
  subprocess looks like. Worth remembering for any credential-derived rule.
- **`ruff check .` passed locally while the flake gate failed.** The devshell ruff
  had a cached verdict; the flake's copy-the-tree run did not. The gate is the
  authority - and the reason the two disagreed at all was that the new files were
  UNTRACKED, so nix (which copies git-tracked files for a dirty tree) had not seen
  them. `git add -A` before `nix flake check` is not optional in a sprout.
- **mypy checks `examples/`.** The example needed the full `AgentBackend` protocol
  (`read_context`, `delete_session`) and the `Backend.MOCK` enum, which is a good
  forcing function: an example that does not satisfy the real protocol is not
  demonstrating the real thing.

## What the review found (round 1, R1.1)

The one finding worth the review's cost was a deadlock this diff introduced. Both
refusals that protect a pending decision - the orchestrator cannot message a
BLOCKED agent, and cannot acknowledge its signal - were keyed on the outcome STATE.
Nothing clears that state when a proposal simply EXPIRES: `approve` and `deny` clear
it by resuming the agent, and an expiry resumes nobody. So one proposal the operator
never answered left the host agent permanently unreachable and unacknowledgeable,
with the approval that would have freed it no longer approvable - every route out
closed at once, and delegation to the host agent broken for the life of the process.

Both refusals now ask `HostApprovalService.live_for_agent`, which runs the record
through the same `_refuse_undecidable` check the decision path uses, so the guard
and the decision cannot disagree about whether an approval is still live. The
acknowledge policy moved out of `AgentStore` (which knows nothing about proposals)
into the route.

The generalisable shape: a guard keyed on a STATE needs something that clears that
state on every path out, and "the happy path clears it" is not that. Here the
missing path was the one where nobody acts at all - which is also the one nobody
writes a test for.

## Self-reflected feedback

- **A rule derived from a field's NAME is a hypothesis until run against real
  data.** `reversal.possible` sounded like exactly the right signal and was wrong
  for the most common action on the box. The plan gate had no way to catch that -
  but writing the one-way test FIRST, before wiring the rule into the route, would
  have surfaced it in a minute instead of via five unrelated red tests.
- **Look for the state you need before adding one.** `AgentState.BLOCKED` was
  sitting in `enums.py` with the exact docstring for the job. The plan invented a
  marker anyway. Grepping the enum for the concept (not the word) is cheap.
- **Deferring the surfaces was the right cut.** Building the core with two callers
  in mind - and pinning "the web route owns no decision rule of its own" with a
  test that drives the SERVICE with a Telegram-shaped actor - means the next task
  adds a translator rather than a second gate. That test is the one that will fail
  loudest if the next task takes a shortcut.
