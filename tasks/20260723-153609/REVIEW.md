# Review: SC1 sub-agent request_input-when-blocked steering

- TASK: 20260723-153609
- BRANCH: feat/sc1-agent-steering
- DATE: 20260723
- REVIEWER: out-of-context agent (round 1)

## Round 1 - VERDICT: APPROVE

Reviewed the single tip commit against the actual code (not just the diff).
Targeted tests (`tests/test_agent.py tests/test_sessions.py`) green.

### Findings

1. [verified-correct] Role gating in `_steer` mirrors `_mcp_overrides` exactly:
   both use `not agent_tools_enabled -> nothing`; `is_orchestrator -> orchestrator
   surface / STEERING_PREAMBLE`; `agent_id -> agent surface /
   AGENT_STEERING_PREAMBLE`; else nothing. A turn gets the agent preamble on
   precisely the same condition it gets the `request_input` tool. Orchestrator
   never gets the wrong preamble (its branch returns first, ignoring `agent_id`);
   a toolless claude sub-agent (`agent_id=""`) is never steered.
2. [verified-correct] `_stream_app_server` threads a real `agent_id` keyword param
   into `_steer` at the turn/start call site (and to `_mcp_overrides`); upstream
   `CodexBackend.stream` forwards it, so a real sub-agent id reaches `_steer` at
   runtime.
3. [verified-correct] `strip_steering` removes the new block: shared
   `_STEER_OPEN`/`_STEER_CLOSE` sentinels, non-greedy DOTALL regex, no bracket
   sentinels in the body.
4. [verified-correct] Tests assert the right things and cover the edge cases:
   agent-gets-preamble, orchestrator-ignores-agent_id, tools-disabled-with-agent_id,
   toolless-claude, and the sessions strip assertion.
5. [nit, non-blocking] The exclusivity of the agent branch relies on the
   `is_orchestrator` branch returning first rather than an explicit
   `not is_orchestrator` guard. Correct as written; a note for any future reorder.
   No change requested.

No ASCII/style issues, no correctness bugs, no comment-vs-code drift. Ship it.
