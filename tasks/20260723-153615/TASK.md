# SC2: orchestrator comms steering (poll pending_agents, answer via message_agent, acknowledge)

- PRIORITY: 37
- TAGS: spike, agents, backend
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As the orchestrator, I want my turn prompt to tell me to check for blocked
sub-agents and answer them, so the comms loop actually closes even with `auto_wake`
off (the poll path) - today my steering says nothing about `pending_agents` /
`message_agent` / `acknowledge`.

## Context (grounded)

`STEERING_PREAMBLE` (`scufris/sessions.py`) only points at the host tools; it never
mentions the agent-comms tools. Per the gate decision (SPIKE),`auto_wake` stays OFF
by default, so the orchestrator must POLL to find WAITING agents - and it must be
STEERED to do so (the codex-steering lesson). The tools exist: `pending_agents` /
`acknowledge` (BC3) and `message_agent` (BC2, resumes a sub-agent's session).
Spike: `tasks/20260723-153339/SPIKE.md` (SC2).

## Steps

- [x] Extend the orchestrator steering (`STEERING_PREAMBLE`, or an added comms
      clause it composes with) with the protocol: "At the end of a turn, call
      `pending_agents` to find sub-agents that need you; for each, answer it via
      `message_agent` (this resumes its session) and then `acknowledge(id)` so it
      stops pending."
- [x] Keep it in the orchestrator-only steering path (`_steer` `is_orchestrator`);
      it must NOT leak into a sub-agent's preamble (SC1 owns that).

## Definition of Done

- An orchestrator turn prompt includes the `pending_agents` -> answer via
  `message_agent` -> `acknowledge` instruction; a sub-agent turn does not.
  (test: `test_steer_orchestrator_gets_comms_protocol`)
- End to end with a real backend: a sub-agent blocks (calls request_input), and on
  the next orchestrator turn the orchestrator polls `pending_agents`, answers via
  `message_agent`, and `acknowledge`s it - the sub-agent resumes with the answer.
  (manual)
- `ruff check .`, `mypy`, `python -m pytest` green. (cmd: `python -m pytest`)

## Notes

- Complements SC1 (sub-agent side) and BC3/BC4 (the notify mechanism the
  orchestrator is now told to use). `auto_wake` (BC4) remains the opt-in push
  alternative to this poll.
- Lesson: `codex-tool-choice-only-steers-via-the-turn-prompt`.
