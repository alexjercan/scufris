# Reduce Telegram to a chat and host-approval surface

- PRIORITY: 93
- TAGS: feature, v0.2.0, lane2, telegram
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -
- PARENT: 20260801-154211
- DEPENDS ON: 20260804-182223

## Story

As the operator, I want Telegram reduced to a conversation and host-approval
surface with no agent operations, so that the demolition in Lane 8 removes the
orchestrator stack without Telegram failing to import.

Scheduled BEFORE any deletion. This is the enabling task, not the reconnection -
that is Lane 8.

## Steps

- [ ] Break the module-scope coupling in `scufris/telegram/wiring.py`. It
      imports `agent_diagnostics`, `agent_store.AgentStore`, `config.Settings`,
      `env_bridge`, `health.AgentHealth`, `mcp_models.AgentTool` and
      `orchestrator.OrchestratorTurnService` at module scope, so deleting the
      orchestrator makes Telegram fail to IMPORT rather than answer politely.
- [ ] Remove the agent-operation surface rather than stubbing it. An agent
      command that returns "not available" is a surface to maintain; the
      reduction is meant to shrink the package.
- [ ] Keep host approvals and the conversation working throughout. The operator
      uses this surface daily and it must not go dark for the rest of the
      release.
- [ ] Prove the decoupling with an import test that fails if a root agent
      module reappears at module scope.

## Definition of Done

- Telegram imports with the orchestrator modules absent from the import graph
  (test: `test_telegram_imports_without_the_agent_stack`).
- No module under `scufris/telegram/` imports an agent, orchestrator or health
  module at module scope
  (test: `test_telegram_has_no_agent_stack_import`).
- The host approval path still works end to end
  (cmd: `python -m pytest tests/test_examples.py`).

## Notes

- Its only HARD constraint is that it precede the `agents` and `flow` deletions
  in Lane 5. It sits in Lane 2 for coherence with the approval card; the epic
  records that it should be MOVED rather than allowed to become Lane 5's
  blocker if Lane 2 slips.
- Depends on the host approval decoupling, because that changes the interface
  the Telegram card is written against.
- Lane 2 of `tasks/20260801-154211/TASK.md`.
