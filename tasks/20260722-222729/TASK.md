# T3: prune the MCP surface (drop tatr_* tools; keep host tools orchestrator-scoped; update steering/tests)

- STATUS: OPEN
- PRIORITY: 34
- TAGS: spike,telegram,agent,mcp

## Goal

Prune the MCP tool surface for the orchestrator-only world. DROP `tatr_ls`,
`tatr_show`, `tatr_new` - `tatr` is a skill the orchestrator runs via `Bash`,
so a dedicated MCP wrapper is redundant once the server is orchestrator-scoped.
KEEP `host_stats`, `disk_usage`, `list_processes`, `list_agents`,
`agent_status` (now orchestrator-only via T1, joined by T2's control tools).
Update the tool-steering preamble and any docs/tests that name the removed
tools.

## Steps

- [ ] Remove the `tatr_ls`, `tatr_show`, `tatr_new` `@mcp.tool()` handlers and
      their helpers/constants (`_TATR_SORTS`) from `scufris/mcp_server.py`;
      update the module docstring that calls `tatr_new` "the one write tool".
- [ ] Edit `STEERING_PREAMBLE` in `scufris/sessions.py`: drop `tatr_ls`,
      `tatr_show`, `tatr_new` from the tool list and remove the "For tatr tasks
      or the backlog use the tatr_* tools" sentence. Keep the host-tool steering.
      (The preamble is already orchestrator-only after T1's `_steer` gating.)
- [ ] Grep for any other references to the removed tools and fix them
      (`grep -rn "tatr_ls\|tatr_show\|tatr_new" scufris/`).
- [ ] Update `tests/test_mcp_server.py`: remove the `tatr_*` cases and drop those
      names from the registration-set assertion; confirm the host tools and the
      T2 control tools remain.

## Definition of Done

- No `tatr_ls`/`tatr_show`/`tatr_new` tool remains in the server or the steering
  preamble.
  (cmd: `grep -rn "tatr_ls\|tatr_show\|tatr_new" scufris/`)
- The registration set is exactly the host tools + observe tools + T2 control
  tools (no tatr).
  (test: `` `test_registered_tools_after_prune` ``)
- `nix flake check` is green. (cmd: `nix flake check`)

## Notes

- Spike: tasks/20260722-221359/SPIKE.md (Q4).
- Depends on: T1.
- Caveat: `tatr_new` wrote from OUTSIDE the model's read-only sandbox; via Bash
  the orchestrator needs a write-capable permission mode to create tasks (SPIKE
  open question - confirm the orchestrator's default mode).
- Update `tests/test_mcp_server.py` tool-set assertions and
  `agent.py` STEERING_PREAMBLE references to tatr tools.
- spike-seeded; plan into steps with /plan before /work.
