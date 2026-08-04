# Reduce Telegram to a chat and host-approval surface

- PRIORITY: 93
- TAGS: feature, v0.2.0, lane2, telegram
- KIND: TASK
- ACTIVITY: UNDERSTANDING
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

- [ ] Break the module-scope coupling. Corrected 2026-08-04 after review: it is
      in FIVE modules, not just `wiring.py` as this record and the epic both
      said - `wiring.py:37-47`, `contracts.py:17-21`, `turn.py:22`,
      `render.py:38-40`, `orchestrator.py:20-21`.
- [ ] Treat the target as NO ROOT IMPORT AT MODULE SCOPE, not "no agent,
      orchestrator or health import". `config.Settings` and `env_bridge` are
      neither, and they are the ones that make the carve impossible: after the
      move they become `scufris_telegram -> scufris` while the graph already has
      `scufris -> scufris_telegram`, and the cycle check fails the suite.
      Inverting those to injected values is the real work here.
- [ ] Remove the agent-operation surface rather than stubbing it. An agent
      command that returns "not available" is a surface to maintain; the
      reduction is meant to shrink the package.
- [ ] Keep HOST APPROVALS working throughout. Corrected after review: keeping
      the CONVERSATION working is not deliverable and this step used to claim
      both. Telegram's conversation IS the orchestrator turn - `contracts.py:27`
      and `orchestrator.py` build the message callbacks from
      `OrchestratorTurnService` - and the new conversation is not connected
      until Lane 8. So either the agent stack stays (defeating the task) or
      Telegram chat is dark from here to Lane 8. Accept the dark period
      explicitly in `DECISION.md`; it is an operator-visible regression across
      several lanes and it should be agreed, not discovered.
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
