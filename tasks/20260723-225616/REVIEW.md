# Review: Read-only per-project skills+tools discovery + endpoint

- TASK: 20260723-225616
- BRANCH: feature/project-capabilities

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

- [ ] R1.1 (NIT) scufris/app.py:1587 - The endpoint canonicalizes with
  `read_project_capabilities(project.cwd, canonical_backend(agent.backend))`,
  and `_sources_for` inside the module canonicalizes again. The double-fold is
  idempotent (harmless); passing `agent.backend` raw would be equally correct
  and less surprising since the module owns canonicalization. No change
  required.
  - Response: Acknowledged; left as-is. Canonicalizing at the boundary is a
    deliberate belt-and-suspenders - a future caller of the endpoint helper
    that forgets to fold still gets correct behavior. Idempotent, no cost.
- [ ] R1.2 (NIT) scufris/project_capabilities.py:171 - `_tool_from_spec` prefers
  `command` over `url` when a spec has both, silently dropping the url in the
  description; worth a comment if precision matters later. No change required.
  - Response: Acknowledged; left as-is. A well-formed MCP entry is stdio XOR
    remote; a spec with both is malformed and the command-wins summary is a
    reasonable best-effort. `kind` still reflects the transport.

Verification (in-session supplement): Independently re-derived the load-bearing
claim - the pre-existing suite failure
`test_agent_config_omits_builtin_server_when_tools_disabled` fails identically on
master in isolation and is caused by that test constructing `Settings(...)`
without an isolated `state_dir` (so it reads the real
`~/.local/state/scufris/settings.json`, whose default profile had
`agent_tools_enabled: true`); it is untouched by this diff and filed as task
20260723-233337. Full check gate re-run in the worktree: ruff format --check,
ruff check, mypy scufris all clean; `pytest` green except that one deselected
pre-existing failure. New tests are behavioral (fail if the module is deleted);
no existing test weakened; DECISION.md records the provider-aware paths + the
read-only scope and is indexed in GOAL.md. Two NITs left to implementer
discretion; neither blocks. No open `manual:` DoD items on this task.
