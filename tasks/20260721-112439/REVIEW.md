# Review: B5a reserved orchestrator agent record

- TASK: 20260721-112439
- BRANCH: feature/orchestrator-reserved-record

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Both suites ran green in the worktree (ruff + mypy 35 files + pytest; web
npm run ci, webpack + vitest). Scope guards respected (Agent protocol +
landing /api/chat* untouched). One NIT, addressed.

Verified by the reviewer:
- Synthetic record built from settings, prepended in list(), returned by get();
  `_persist` only serializes `_agents`, so the orchestrator is never written to
  agents.json; `_load` never touches it. delete/update raise ReservedAgent;
  create refuses the reserved id (`_slugify("Orchestrator")` == "orchestrator").
  mark_running/mark_finished mutate only the in-memory ints.
- Projectless: `_require_agent_project` returns None only for empty project_id
  (still 422s a real-but-missing project); `_launch_agent_turn` uses cwd=None;
  no other caller derefs project.cwd for the orchestrator; run_agent 422s it for
  lacking a goal (correct - it is chatted). Endpoints: DELETE 403, PATCH 409,
  GET 200 with project_id "".
- Tests meaningful; the 4 empty-list tests narrowed (not weakened). Frontend
  hides delete + Settings for the reserved id, shows "server dir".
- Close-out matches the code incl. the deferred editable-config (update->409).

- [x] R1.1 (NIT) mcp_server.py:230 - the `if not agents: return "no agents
  configured."` branch is now unreachable (the list always has the orchestrator).
  - Response: fixed - removed the dead branch (it is dead permanently, not just
    scaffolding) with a comment noting the list is never empty.
