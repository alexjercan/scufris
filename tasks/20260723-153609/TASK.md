# SC1: sub-agent request_input-when-blocked steering (teach sub-agents to signal)

- STATUS: CLOSED
- PRIORITY: 38
- TAGS: spike, agents, backend
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

As a sub-agent, I want to be TOLD (in my turn prompt) to call `request_input` when
I am blocked, so I actually signal the orchestrator instead of guessing or stopping
silently - the tool exists (BC2) but nothing steers me to use it.

## Context (grounded)

`_steer` (`scufris/agent.py`) prepends `STEERING_PREAMBLE` ONLY for the
orchestrator (`is_orchestrator`); every non-orchestrator turn gets the prompt
unchanged. Per `codex-tool-choice-only-steers-via-the-turn-prompt`, codex won't
call `request_input` off its description alone - the instruction must ride the
turn prompt. Sub-agents get the `request_input` tool only when they have the
agent-role scufris server (`_mcp_overrides` `elif agent_id`, codex + tools
enabled), so the steering must match that gate (never a claude sub-agent, never
the orchestrator). Spike: `tasks/20260723-153339/SPIKE.md` (SC1).

## Steps

- [x] Add an agent steering preamble (`sessions.py`, reuse `_STEER_OPEN`/
      `_STEER_CLOSE` so `strip_steering` cleans it): "If you are blocked or need a
      decision/approval you cannot safely make yourself, call
      `request_input(question)` with a clear question and STOP; do not guess - the
      orchestrator will answer and resume you."
- [x] Thread `agent_id` into `_steer` and pick the preamble: `is_orchestrator` ->
      the host-tools preamble (unchanged); else `agent_id` + `agent_tools_enabled`
      -> the new agent preamble; else the prompt unchanged. `_stream_app_server`
      passes `agent_id` to `_steer`.
- [x] Confirm `strip_steering` removes the agent block from recorded messages
      (same markers -> already covered; add an assertion).

## Definition of Done

- A sub-agent turn (`is_orchestrator=False`, `agent_id` set, tools enabled) has the
  `request_input`-when-blocked instruction in its prompt; the orchestrator's turn
  does NOT (it keeps the host-tools preamble); a turn with no tool (no `agent_id`
  or tools disabled) gets neither.
  (test: `test_steer_agent_gets_request_input_preamble`)
- `strip_steering` removes the agent block. (test: covered)
- A real blocked codex sub-agent calls `request_input` (steering works).
  (manual: run a sub-agent on a task that requires an approval it lacks; confirm it
  calls request_input rather than guessing/stopping silently)
- `ruff check .`, `mypy`, `python -m pytest` green. (cmd: `python -m pytest`)

## Notes

- The missing half of BC2 (`tasks/20260723-094303`). Composes with SC2
  (orchestrator side) and BC3/BC4 (notify).
- Lesson: `codex-tool-choice-only-steers-via-the-turn-prompt`.
