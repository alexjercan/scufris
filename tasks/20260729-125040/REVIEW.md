# Review: Add the host operator agent and the approval decision core

- TASK: 20260729-125040
- BRANCH: feat/host-operator-agent

## Round 1

- VERDICT: REQUEST_CHANGES
- REVIEWER: in-session (no out-of-context mechanism available - subagents are
  disabled in this session, so the round-1 default could not be used. Recorded as
  the exception the review skill allows; the findings below were produced by
  re-reading the diff against TASK.md and by RUNNING probes rather than by
  reasoning about the implementation, and the load-bearing one is pinned by a
  throwaway test whose output is quoted.)

What was verified rather than taken on trust:

- `python -m pytest`: 834 passed. `nix flake check`: all 4 checks pass (ruff,
  mypy over 108 files including `examples/`, pytest, records).
  `nix build .#scufris .#web`: exit 0. `nix build .#hostd-vm-test`: exit 0 - so
  the new `list_pending` verb is exercised against a REAL root helper on a real
  socket, not only against the injected engine.
- The audience split was checked from both ends and by listing the live
  registries: `scufris` advertises 30 tools, `host` 20, and
  `scufris & host == INSPECTION` exactly, with `ACTIONS` on `host` alone.
- Every DoD-named test exists and fails for the right reason when its subject is
  broken: the one-way test was watched failing (422) before the acknowledgement
  was added, and the five pre-existing host-action API tests going red at once is
  what surfaced the confirmation-rule error recorded in `DECISION.md` section 6.
- Both `examples/host_agent.py` paths were run end to end and their output read.

- [x] R1.1 (MAJOR) scufris/app.py:2447 (the `/chat` BLOCKED guard) and
  scufris/agent_store.py:1015 (`acknowledge` refusing BLOCKED) - an abandoned
  proposal leaves the host agent PERMANENTLY unreachable, so one undecided
  proposal breaks delegation to it for good. The guard and the acknowledge refusal
  both key on the outcome STATE, which nothing clears when a proposal simply
  expires: `deny`/`approve` clear it by resuming the agent, but an expiry resumes
  nobody. Probed directly - propose as the host agent, expire the proposal, then:
  `POST /api/agents/host/chat` (machine credential) -> `409 ... waiting for the
  OPERATOR`, `POST /api/agents/host/acknowledge` -> `{"acknowledged": false}`,
  `POST /api/host/actions/<id>/approve` -> `409 ... has expired`, and
  `/api/agents/pending` still reports `blocked`. Every route out is closed at
  once. Suggested change: make both refusals depend on the approval being LIVE
  rather than on the state alone - add
  `HostApprovalService.live_for_agent(agent_id)` returning the pending,
  unexpired record whose `requester.agent` is that agent, gate the chat refusal on
  it, and move the acknowledge policy out of the store (which knows nothing about
  proposals) into the route, gated on the same check. Pin it with a test that
  expires a proposal and asserts the agent becomes both messageable and
  acknowledgeable.
  - Response: fixed. `HostApprovalService.live_for_agent` answers "is this agent
    waiting on a decision that can still be made" by running the record through
    `_refuse_undecidable` - the same check the decision path uses, so the guard and
    the refusal cannot disagree. The chat guard and the acknowledge policy now both
    read it, the acknowledge policy moved out of `AgentStore` (which knows nothing
    about proposals) into the route, and the 409 now names the action id so the
    orchestrator can report WHAT is waiting. Pinned by
    `test_an_undecided_approval_does_not_strand_the_agent`, which asserts both
    refusals hold while the approval is live and both release once it expires.
- [x] R1.2 (MINOR) scufris/app.py:1495 - the `/confirmation` route's docstring
  says the requirement "is also carried inline on every record in the queue
  listing", and it is not: `HostActionRecord` has no such field, so a queue
  render would need one request per row to know whether a row is one-way. Either
  correct the sentence or make it true. Making it true is better - the surfaces
  (20260730-104520, 20260730-104524) both render per-row risk - which needs
  `Confirmation` and `confirmation_for` to live in `host_actions` (where the
  record is) so the record can expose a computed field without importing
  `host_approvals` and creating a cycle.
  - Response: made true. `ConfirmationStyle`, `Confirmation` and `confirmation_for`
    moved to `host_actions` (next to the record they describe), and
    `HostActionRecord.confirmation` is a pydantic computed field, so every row of the
    queue listing carries the requirement inline and the `/confirmation` route is now
    only for a client holding one id. Its docstring says that instead.
- [x] R1.3 (MINOR) README.md:160 - "Post a ref to `/api/host/config/changes` (or
  let the orchestrator do it with `propose_nixos_change`)" is now false: that tool
  moved off the orchestrator's server in this diff. The doc-surface sweep caught
  the other three mentions and missed this one. Suggested change: name the host
  agent as the caller, matching the sentence added at README.md:110.
  - Response: fixed - the README now names the host agent and states that the tool is
    on its server, not the orchestrator's.
- [x] R1.4 (MINOR) tests - the audience change is pinned at the module level
  (`scufris_mcp_servers`, `_mcp_overrides`, the live registries) but not at the
  two ENDPOINTS that report an agent's tools to the operator:
  `GET /api/agents/{id}/tools` and `GET /api/agent/tools`. Those are what the
  settings page shows, and the ledger's
  `tool-reachable-by-two-runners-needs-a-test-per-runner` is exactly this shape -
  the listing can drift from the wiring while both look right in isolation.
  Suggested change: one test asserting `/api/agents/host/tools` includes the
  propose tools and `/api/agent/tools` (the console, orchestrator-scoped) does
  not.
  - Response: added `test_the_tool_listings_report_the_audience_they_wire`, which
    drives both endpoints plus a regular project agent's, and asserts the SERVER each
    tool is reported under - so a tool listed from the wrong server fails too.
- [x] R1.5 (NIT) scufris/mcp_server.py:855 (`update_agent`) and :890
  (`delete_agent`) - `_reject_orchestrator` gives the orchestrator a clear
  tool-level refusal, while the host agent falls through to a 409/403 from the
  API. The outcome is correct either way; the message is just less useful.
  Suggested change: broaden the helper to the reserved set and rename it
  `_reject_reserved`.
  - Response: done - `_reject_reserved` refuses the whole reserved set, and
    `test_agent_write_tools_reject_the_reserved_agents` now covers both ids (and
    asserts the set itself, so a third reserved agent added later is not silently
    left out).

Pending user checks (not resolved by this review):

- manual: asking the orchestrator to restart a service or change the config
  reaches the host agent, and the pending decision is visible and honest about
  what it would do.

## Round 2

- VERDICT: APPROVE
- REVIEWER: in-session (the same exception as round 1: subagents are disabled in
  this session. Each fix was re-verified against the new diff by running its pin,
  and R1.1's was re-derived by replaying the original probe rather than by reading
  the patch.)

All five round-1 findings are resolved and ticked:

- R1.1 re-verified by replaying the probe that found it: while the approval is live
  the chat is refused (409, now naming the action) and the acknowledge returns
  `false`; once the window closes the same two calls return 200 and `true`, and the
  stale row leaves `/api/agents/pending`. Reverting the fix fails the pin - the
  state-keyed version returns `false` forever.
- R1.2 verified by reading a queue row: `GET /api/host/actions` now carries
  `confirmation.style`, `risk_label`, `undo`, `no_undo` and `acknowledge` per row.
- R1.3 verified by grep: no doc surface now attributes a propose tool to the
  orchestrator.
- R1.4 and R1.5 verified by their new tests, which fail if the audience wiring or
  the reserved set regresses.

Gate after the fixes: `python -m pytest` 839 passed; `nix flake check` all four
checks pass; `nix build .#scufris .#web` exit 0; `nix build .#hostd-vm-test` exit 0
(run before the fixes, which touch no helper code).

Pending user checks (not resolved by this review):

- manual: asking the orchestrator to restart a service or change the config
  reaches the host agent, and the pending decision is visible and honest about
  what it would do.
