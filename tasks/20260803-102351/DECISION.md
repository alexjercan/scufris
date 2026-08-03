# Decision: `fork_seed` lives on a subclass in the trap file, not on the shared fake

- DATE: 2026-08-03
- STATUS: ACCEPTED
- TASK: 20260803-102351
- TAGS: tests, refactor, backend

## Context

Step R2.1a asked for `fork_seed` on `FakeRunService`
(`tests/test_orchestrator_routers.py`), so the agent-run trap could drive
`/fork`. That file is 897 lines against the 900-line test cap enforced by
`scripts/check_file_size.py`, so the method's six formatted lines pushed it to
907 and turned `tests/test_check_file_size.py` red.

`ALLOWLIST` is a ratchet - the guard's docstring says a new oversized file is
"the failure the guard exists to report" and forbids new entries - so silencing
it was not available. The plan named a file it had not measured; the change
itself was never in question.

## Decision

`ForkingRunService(FakeRunService)` carries the method in
`tests/test_agent_run_router.py`, and `RunTrapRig` constructs it.

The signature still matches `AgentRunService.fork_seed`
(`scufris/orchestrator/runs.py:468`) and the call is still recorded on
`self.calls` as `("fork_seed", (agent.id, session_id, message_index, text))`, so
R2.1a's substance is unchanged - only its file.

## Alternatives considered

- **Add it to the shared `FakeRunService` as planned.** Blocked: 897/900 leaves
  three lines against a six-line method.
- **Allowlist the oversized file.** Rejected: the guard is a ratchet by design
  and its docstring forbids new entries.
- **Split `test_orchestrator_routers.py` along its three rigs** (project,
  agent-run, agent-record). Correct, and the real fix - but a restructure well
  past this task's scope. Wants its own task.

## Consequences

This is in idiom rather than a workaround: `test_agent_run_router.py` already
redefines `FullDiagnostics` for exactly this reason - its module docstring says
"only the diagnostics fake is redefined, because the run surface asks it for
`health`, `tools` and `mcp` as well as the three the shared one answers".
`/fork` is the same shape: it belongs to this file's surface and nothing else
asks for it. `test_orchestrator_routers.py`'s own fork test
(`test_the_orchestrator_cannot_revert_fork_through_the_agent_route`) 409s on the
orchestrator id before the route ever reaches `fork_seed`, so the shared fake
does not need the method.

The cost: the run-service fake is now split across two files, and a reader
looking for its full surface must follow the subclass. The 897/900 headroom
stays load-bearing - the next addition to any of those three rigs hits the same
wall, and should split the file rather than repeat this move.
