# Review: U1 - orchestrator as a first-class hidden, editable agent

- TASK: 20260721-234558
- BRANCH: feature/orchestrator-first-class

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (fresh subagent, no sight of the implementing session;
  ran both suites itself and re-derived the load-bearing claims - the list-exclusion
  has no broken consumer, and the settings-store edit round-trip + session-clear)

Clean, well-scoped diff implementing spike recommendation B1. Both suites green:
backend `ruff`+`mypy`+`pytest` (281 passed) and web `npm run ci` (vitest + webpack).
The three `list()` consumers are all accounted for (`/api/agents` and the mcp
agent-list intentionally exclude the orchestrator; the frontend now shows a real
"no agents yet" empty state instead of the orchestrator masking it). The
orchestrator stays resolvable via `get()` for every per-agent endpoint. The
settings-store edit round-trip (backend/model-by-effective-backend/permission_mode),
the 403/422 paths, the backend-change session clear (rebuild key), and the
WRITABLE_KEYS/AgentConfigUpdate sync were all verified.

- [ ] N1 (NIT) app.py `_update_orchestrator` - name/description/goal/task_id are
  silently dropped for the orchestrator (a caller PATCHing `name` gets 200 with no
  change). Optional explicit comment.
  - Response: No change - the docstring already states "Name/description/goal/
    task_id are fixed for the orchestrator and ignored", so it is explicit. The
    orchestrator's name/description are fixed constants by design.
- [ ] N2 (NIT) the SPIKE's open question "is the orchestrator always manual today?"
  is now resolved (defaults MANUAL, editable) - the spike prose is stale, not the
  diff.
  - Response: No change - the spike is append-only history; the resolution is
    recorded here and in the code.

No pending manual DoD items (backend-only task; the DoD is machine-proved).
