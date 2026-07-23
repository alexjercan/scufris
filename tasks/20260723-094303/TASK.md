# BC2: request_input sub-agent callback tool (needs-input signal; role-scoped tools)

- STATUS: OPEN
- PRIORITY: 38
- TAGS: spike,agents,backend,mcp

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

- [ ] Generalize the `is_orchestrator` gate in `_mcp_overrides`
      (`agent.py:153-194`) into a ROLE/audience (`orchestrator` vs `agent`); tag
      each existing tool `orchestrator` (control/observe/host) and mark
      `request_input` as `agent`. Register the one scufris server for sub-agents
      too, passing the role via env; expose only the caller-role's tools at
      startup (allowlist-by-role, reusing the `codex-per-server-env-filters`
      machinery). No second server.
- [ ] Implement `request_input(question)` in `mcp_server.py`: HTTP-POST a
      needs-input signal to a new app endpoint (e.g. `POST
      /api/agents/{id}/request_input`) carrying the structured question payload;
      the caller's agent id comes from the same env convention T2 uses.
- [ ] App endpoint sets the agent's `WAITING` outcome (BC1) with the question;
      returns immediately (fire-and-forget - do NOT block the sub-agent turn
      awaiting an answer, per the SPIKE open-question resolution).
- [ ] Structured question payload shape (list of questions / free-form text +
      optional context); document it.
- [ ] Track the claude `--mcp-config` parity gap as a follow-up (do not block
      v1 on it).

## Definition of Done

- A sub-agent calling `request_input("merge to master?")` leaves a `WAITING`
  outcome carrying the question; the orchestrator role still gets the full scufris
  surface, the agent role gets ONLY `request_input` (no control tools).
  (test: `test_request_input_sets_waiting_outcome`; test:
  `test_agent_role_exposes_only_request_input` - agent vs orchestrator audiences
  differ)
- `request_input` returns immediately (does not block the sub-agent turn).
  (test: covered above)
- `ruff check .`, `mypy` touched files, `python -m pytest` green from the
  worktree. (cmd: `python -m pytest`)

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
