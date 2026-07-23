# Review: role-scoped per-agent tools endpoint + settings panel

- TASK: 20260723-193216
- BRANCH: feat/agent-tools-role-scoped
- DATE: 20260723
- REVIEWER: out-of-context agent (round 1)
- VERDICT: APPROVE

## Round 1 - VERDICT: APPROVE

Verified against the real code; reviewer ran ruff + mypy (no regressions) + backend
pytest + the web suite (162 passed).

### Findings (all non-blocking, verified OK)

1. `role_tool_names` matches the original `apply_role` scoping byte-identically:
   agent role `names & _AGENT_ROLE_TOOLS`; `apply_role` consumes it via
   `names - keep`, and `names - (names & X) == names - X`. The intersection form is
   safer for the read-only listing (won't surface an unregistered role tool). No
   regression to the spawned-server path.
2. Endpoint: `_require_agent` resolves the orchestrator and 404s unknown; role by
   `agent.id == ORCHESTRATOR_ID`, gated on `_agent_has_scufris_mcp`
   (= `_agent_is_codex`). Codex sub-agent -> `[request_input]`, mock -> `[]`,
   codex orchestrator -> full minus `request_input`. Confirmed by the new test.
3. Route `/api/agents/{id}/tools` does not shadow `/api/agents/pending` (static,
   declared first) or `/api/agents/{id}`. No shadowing.
4. `_as_agent_tool` refactor preserves the exact output shape and
   `enabled = name not in disabled` semantics; `/api/agent/tools` console unchanged.
5. Frontend: `agentToolsPanel` renders only for non-orchestrator; fetch degrades
   (`maybe -> null -> []`); `escapeHtml` on name AND description (no XSS); both
   `AgentSettingsData` constructors supply the new field.
6. The two-endpoint split is sound: the console runs all ~18 in-process; the
   per-agent endpoint reflects what the agent's spawned server advertises. Distinct
   questions, made explicit in docstrings/CHANGELOG.
7. Orchestrator on a non-codex backend: `/api/agents/orchestrator/tools` -> `[]`,
   but the orchestrator page never calls it (uses the global console). Works.

### Nits (cosmetic, no action)
- Populated card title `Tools (n)` vs empty card `panel("tools", ...)` lowercase.
- A fetch failure is indistinguishable from a genuinely empty set (both "none") -
  acceptable graceful degradation.

No non-ASCII, no comment/code drift, no correctness bugs. Ship it.
