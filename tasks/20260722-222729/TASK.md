# T3: prune the MCP surface (drop tatr_* tools; keep host tools orchestrator-scoped; update steering/tests)

- STATUS: CLOSED
- PRIORITY: 34
- TAGS: spike, telegram, agent, mcp
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Goal

Prune the MCP tool surface for the orchestrator-only world. DROP `tatr_ls`,
`tatr_show`, `tatr_new` - `tatr` is a skill the orchestrator runs via `Bash`,
so a dedicated MCP wrapper is redundant once the server is orchestrator-scoped.
KEEP `host_stats`, `disk_usage`, `list_processes`, `list_agents`,
`agent_status` (now orchestrator-only via T1, joined by T2's control tools).
Update the tool-steering preamble and any docs/tests that name the removed
tools.

## Steps

- [x] Removed the `tatr_ls`, `tatr_show`, `tatr_new` `@mcp.tool()` handlers and
      `_TATR_SORTS` from `scufris/mcp_server.py`; rewrote the module docstring
      (now describes host / observe / control groups and says tatr is intentionally
      not here).
- [x] Edited `STEERING_PREAMBLE` in `scufris/sessions.py`: dropped the tatr tools
      from the list and the "For tatr tasks..." sentence; kept the host-tool
      steering and noted it rides only the orchestrator's turns.
- [x] Grepped for references and fixed them: `config.py` comments (removed
      `tatr_*` / `["tatr_new"]` examples), `tests/test_app.py` (tools-endpoint tests
      re-pointed at surviving tools), `tests/test_agent.py` (disabled-tools example),
      and a `web/src/agent-view.test.ts` render fixture (`tatr_ls` -> `disk_usage`).
      `tests/test_projects.py:_tatr_new` is a local `tatr` CLI helper (not the MCP
      tool) and was left as-is.
- [x] Updated `tests/test_mcp_server.py`: removed the seven `tatr_*` tests and the
      `_new_task` helper (and the now-unused `subprocess`/`time` imports), dropped
      the names from the registration-set assertion, and re-pointed the
      disabled-tools test at `disk_usage`.

## Definition of Done

- No `tatr_ls`/`tatr_show`/`tatr_new` tool remains in the server or the steering
  preamble.
  (cmd: `grep -rn "tatr_ls\|tatr_show\|tatr_new" scufris/`)
- The registration set is exactly the host tools + observe tools + T2 control
  tools (no tatr).
  (test: `` `test_tools_registered` ``)
- `nix flake check` is green EXCEPT the pre-existing mypy red (task 20260720-174021);
  ruff + pytest legs pass and the changed source files add zero mypy errors.
  (cmd: `nix flake check`)

## Notes

- Spike: tasks/20260722-221359/SPIKE.md (Q4).
- Depends on: T1.
- Caveat: `tatr_new` wrote from OUTSIDE the model's read-only sandbox; via Bash
  the orchestrator needs a write-capable permission mode to create tasks (SPIKE
  open question - confirm the orchestrator's default mode).
- Update `tests/test_mcp_server.py` tool-set assertions and
  `agent.py` STEERING_PREAMBLE references to tatr tools.
- spike-seeded; plan into steps with /plan before /work.

## Implementation (close)

Removed the three `tatr_*` MCP tools and `_TATR_SORTS`, trimmed the steering
preamble, and swept every reference (source comments, three test files, one
frontend render fixture). The registration set is now the five host/observe tools
plus T2's five control tools. Verification: ruff + full pytest green (348 tests,
down 7 from the removed tatr tests); mypy clean on the changed source files; the
one touched frontend test passes under vitest. `nix flake check` mypy leg remains
pre-existing-red (task 20260720-174021).

Caveat carried forward (SPIKE Q4 open question): `tatr_new` wrote from OUTSIDE the
model's read-only sandbox, so with the MCP tool gone the orchestrator must run in a
write-capable permission mode to create tasks via Bash. Not addressed here (it is a
config/permission decision, not a code change); flagged in the SPIKE open questions.

Self-reflection: the removal sweep reached further than the plan's Steps named -
config.py comments, test_app.py's tools-endpoint tests, and a web/ render fixture all
referenced the tools. Grepping the worktree up front (per the work-skill removal-sweep
rule) caught most in one pass, including the frontend fixture the Python suite would
never reach. BUT the first sweep used `--include=*.py --include=*.md ...` globs, which
skipped the extensionless `.env.example` - review round 1 (R1.1) caught the stale
`["tatr_new"]` there. Lesson: an absence-proving grep must not be narrowed by
file-extension globs when the string can live in a dotfile/extensionless config; sweep
by path, not by extension, or add the dotfiles explicitly.
