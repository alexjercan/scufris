# Review: Parent-session routing for sub-agent escalations

- TASK: 20260724-132830
- BRANCH: feature/parent-session-routing

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Check suite (out-of-context reviewer, re-confirmed in-session): pytest 448
passed, ruff clean, mypy clean, `nix flake check` green; 19 comms/pending/
request_input tests intact; `examples/comms_loop.py` runs green (no-query poll ->
unattributed child still visible). All DoD tests pass; A/B confirmed for the env
injection and the pending filter.

Independently re-derived in-session: the filter drops a child ONLY when a query is
present AND its parent session is non-empty AND differs - so no-query keeps all
(back-compat), unattributed is always kept, own-chat kept, other-chat dropped; no
child is ever invisible to its owner or to all chats.

- [x] R1.1 (MINOR) scufris/mcp_server.py:607-614 - the `pending_agents()` MCP
  table renders only ID/STATE/MESSAGE, but TASK step 5 + DECISION say it
  "annotates each row with its parent chat". The API row carries
  parent_agent_id/parent_session_id, but the tool's rendered table drops it, so
  the operator LLM never sees the attribution - a code/claim mismatch.
  - Response: fixed - added a `PARENT` column to the
    `pending_agents()` table rendering each row's `parent_session_id` (short id,
    "-" when unattributed). Verified in-session
    (test_pending_agents_scopes_to_the_calling_chat extended to assert the parent
    is rendered).
- [ ] R1.2 (NIT) scufris/mcp_server.py:602 - `parent_session_id` interpolated
  raw into the query string; safe today (ids are UUIDs) but `urllib.parse.quote`
  is more robust.
  - Response: fixed - the id is now URL-encoded via
    `urllib.parse.quote`.
- [ ] R1.3 (NIT) scufris/mcp_server.py:648 - `_orch_session_id` does a
  function-local `import os`.
  - Response: left as-is - it deliberately mirrors `_self_agent_id` directly
    above it (same function-local `import os`); matching the neighbour is the more
    consistent choice here.

No BLOCKER/MAJOR findings. Verdict APPROVE; the MINOR + one NIT were addressed as
discretionary follow-ups.
