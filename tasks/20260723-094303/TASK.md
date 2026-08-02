# BC2: request_input sub-agent callback tool (needs-input signal; role-scoped tools)

- PRIORITY: 38
- TAGS: spike, agents, backend, mcp
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As a sub-agent, I want a narrow, notify-only tool to tell the orchestrator "I'm
blocked, here's my question" and end my turn, so the orchestrator can answer later
by resuming my session - instead of the loop stalling because nobody knew I was
waiting.

## Context (grounded)

Sub-agents currently get NO scufris MCP tools: `_mcp_overrides`
(`agent.py:153-194`) registers the whole scufris server ONLY when
`is_orchestrator` (`app.py:1106`). T3 (`tasks/20260722-222729`) was NOT a hard
"sub-agents get nothing" security boundary - it was a CAPABILITY preference
("none of the current tools are useful for sub-agents, and I don't want them
creating/running other agents"). So BC2 exposes `request_input` to sub-agents via
a ROLE-SCOPED tool model (see `DECISION.md`, Option B): ONE scufris server,
generalize `is_orchestrator` into a role/audience (`orchestrator` vs `agent`),
tag each tool with its audience, expose only the caller-role's tools. This
preserves the guarantee T3 cares about (sub-agents cannot reach the control
tools) as an explicit allowlist, without a second server. The MCP server is a
separate subprocess and cannot touch the live supervisor, so the tool reaches
back over the local HTTP API (the T2 control-tool pattern, `127.0.0.1:<port>`).
`request_input` hard-sets the caller's `WAITING` outcome (BC1) with a structured
question payload. Claude sub-agents get no scufris MCP today (`backends.py` never
adds `--mcp-config`), so this is CODEX-FIRST; track the claude parity gap.

Spike: `tasks/20260723-001256/SPIKE.md` (BC2, the chosen signal). GATE DECISION:
explicit callback tool over inference. Tool-exposure decision: `DECISION.md`
(role-scoped, one server - Option B).

## Steps (/plan expands)

- [x] Generalize the `is_orchestrator` gate in `_mcp_overrides` into a
      ROLE/audience (`orchestrator` vs `agent`): `mcp_server.apply_role(role)`
      keeps only `_AGENT_ROLE_TOOLS = {"request_input"}` for the agent role and
      everything else (dropping the agent-only tools) for the orchestrator, called
      in `main()` from `SCUFRIS_AGENT_ROLE`. `_mcp_overrides` registers the ONE
      scufris server for sub-agents too, tagged `SCUFRIS_AGENT_ROLE=agent` +
      `SCUFRIS_AGENT_ID`. No second server (`DECISION.md`, Option B).
- [x] Implement `request_input(question)` in `mcp_server.py`: reads its own id
      from `SCUFRIS_AGENT_ID`, HTTP-POSTs `{question}` to
      `POST /api/agents/{id}/request_input` (the T2 `_api_call` pattern).
- [x] App endpoint (`agent_request_input`) sets the agent's `WAITING` outcome
      (`AgentStore.request_input`, keyed to the current run) with the question and
      returns immediately (`RequestInputResult`); `mark_finished` preserves the
      same-run WAITING so the turn-end DONE does not clobber it.
- [x] Payload shape: a single free-form `question: str` (the model asks one
      question per turn); documented on `AgentRequestInput`. A list/structured
      form was not needed for v1 and would add surface BC3/BC4 do not use.
- [x] Track the claude `--mcp-config` parity gap as a follow-up (see Notes; not a
      v1 blocker). `agent_id` threaded through all four backend `stream`
      signatures; only the codex app-server path forwards it.

## Definition of Done

- A sub-agent calling `request_input("merge to master?")` leaves a `WAITING`
  outcome carrying the question; the orchestrator role still gets the full scufris
  surface, the agent role gets ONLY `request_input` (no control tools).
  (test: `test_request_input_sets_waiting_outcome`; test:
  `test_apply_role_agent_keeps_only_request_input` /
  `test_apply_role_orchestrator_drops_only_the_agent_tools` - agent vs
  orchestrator audiences differ)
- `request_input` returns immediately (does not block the sub-agent turn), and
  its WAITING signal survives the turn-end completion of the same run.
  (test: `test_request_input_records_waiting_outcome`;
  `test_waiting_survives_same_run_completion`)
- `ruff check .`, `mypy`, `python -m pytest` green from the worktree.
  (cmd: `python -m pytest`)

## Notes

- Depends on BC1 (the `WAITING` outcome substrate).
- Tool-exposure decision recorded in `DECISION.md` (role-scoped, one server -
  Option B; T3 reframed as a capability preference, not a security boundary).
- The role model generalizes the `is_orchestrator` plumbing BC3/BC4 already
  assume: BC3's `pending_agents`/`acknowledge` become `orchestrator`-audience
  tools under the same mechanism.
- Lessons: `codex-mcp-register-via-c`, `codex-per-server-env-filters-mcp-tools`
  (how the per-role tool subset is scoped), `absence-grep-must-not-be-extension-scoped`
  (sweep `.env.example` for any tool-list references).
- Spike-seeded (BC2).

## Close record (2026-07-23)

What changed, by layer:
- `mcp_server.py`: new `request_input(question)` tool (reads `SCUFRIS_AGENT_ID`,
  POSTs `{question}` to `/api/agents/{id}/request_input` via `_api_call`); a role
  model - `ROLE_ORCHESTRATOR`/`ROLE_AGENT`, `_AGENT_ROLE_TOOLS = {"request_input"}`,
  `_role()` (from `SCUFRIS_AGENT_ROLE`), and `apply_role(role)` that removes every
  tool outside the role's audience; `main()` calls `apply_role` before
  `apply_disabled_tools`. Module docstring updated from "orchestrator-only" to
  role-scoped.
- `agent.py` `_mcp_overrides`: generalized to `is_orchestrator` OR `agent_id` -
  the orchestrator registers scufris with `SCUFRIS_AGENT_ROLE=orchestrator`; a
  regular agent with an id registers it with `SCUFRIS_AGENT_ROLE=agent` +
  `SCUFRIS_AGENT_ID`. `agent_id` threaded through `_stream_app_server` and all
  four backend `stream` signatures (only the codex app-server path forwards it).
- `agent_store.py`: `AgentStore.request_input(agent_id, question, run_id,
  session_id)` writes a WAITING outcome after an existence check; `mark_finished`
  preserves a same-run, unacknowledged WAITING outcome on a DONE completion
  (refreshing the finalized session id) so the natural turn-end does not clobber
  the needs-input signal - keyed on `run_id` so a WAITING from an earlier run is
  still overwritten, and an ERROR still wins.
- `app.py`: `POST /api/agents/{id}/request_input` (`agent_request_input`) with
  `AgentRequestInput`/`RequestInputResult`; the turn launch threads
  `agent_id=agent.id` to the backend. CHANGELOG Added entry.

Evidence: all layers tested (~15 tests, red-first where behavioral): store
(request_input sets WAITING; same-run completion preserves it; a new run's
completion overwrites a stale WAITING; ERROR wins; deleted-agent raises);
mcp_server (agent role exposes only request_input; orchestrator role drops it;
request_input posts/validates); agent (`_mcp_overrides` agent-role env + id;
orchestrator precedence; disabled -> none); app (endpoint 200/422/404; the turn
threads agent_id to the backend). Suite 360 passed (346 baseline + new); ruff +
mypy clean from the worktree.

Difficulties: (1) `main()` now role-scopes the global `mcp` registry, so
`test_main_configures_logging_and_runs` had to gain the `restore_tool_registry`
fixture or it leaked a trimmed tool set into later tests (caught by three
same-file failures). (2) Threading `agent_id` through the shared `stream` protocol
broke the test doubles in test_backends/test_app that stub `_stream_app_server` /
`FakeBackend.stream` with an explicit signature - each fake gained the kwarg.

The load-bearing design call: `request_input` fires MID-turn, so the immediate
turn-end DONE would clobber the WAITING outcome. Resolved by run-id-keyed
preservation in `mark_finished` rather than a separate store or an acknowledge
dependency - correct in isolation (no coupling to BC3/BC4), and pinned by
`test_waiting_survives_same_run_completion` + `test_stale_waiting_overwritten_by_a_new_run`.

Follow-up (not a v1 blocker): claude sub-agents get no scufris MCP wiring
(`backends.py` never adds `--mcp-config`), so `request_input` is codex-only until
that lands - tracked here for a future task.

Self-reflection: the role model paid off exactly as the DECISION predicted -
generalizing `is_orchestrator` (not adding a second server) meant BC3's
orchestrator-only tools drop in as another `orchestrator`-audience entry with no
new plumbing. Writing the store layer first (with the preservation semantics) let
the mcp/app layers be thin wiring over a already-correct primitive.
