# Spike: teach both sides the comms protocol (steering, not just plumbing)

- DATE: 20260723-153339
- STATUS: RECOMMENDED
- TAGS: spike, agents, backend

## Question

BC1-BC4 built the whole message CHANNEL - `request_input` -> a durable `WAITING`
outcome -> wake bridge (BC4) / `pending_agents` poll (BC3) -> the orchestrator
answers via `message_agent` (resuming the sub-agent's session) -> the sub-agent
continues. But neither MODEL is taught the PROTOCOL. How does a sub-agent KNOW to
call `request_input` when it is blocked, and how does the orchestrator KNOW to
poll, answer, and acknowledge? Make the round-trip actually get USED, cleanly.

## Context (grounded - the gap)

- `STEERING_PREAMBLE` (`scufris/sessions.py`) is about the HOST tools only
  (host_stats/disk_usage/list_processes) and is ORCHESTRATOR-ONLY: `_steer`
  (`scufris/agent.py`) returns the turn prompt UNCHANGED for any non-orchestrator
  turn. So a sub-agent has the `request_input` tool (BC2) but is NEVER told to use
  it.
- Load-bearing lesson `codex-tool-choice-only-steers-via-the-turn-prompt`: codex
  ignores tool descriptions and instructions files for tool CHOICE; the
  instruction must ride the TURN PROMPT (in the live probe a prepended preamble
  moved tool use from 0 to 3 MCP calls). A tool the model is not steered to use is
  not reliably used.
- The orchestrator's steering says nothing about `pending_agents` /
  `message_agent` / `acknowledge` or the wake, so it is not taught to poll,
  answer, or clear.

Conclusion: BC1-BC4 delivered the channel; this spike delivers the missing "teach
the models the protocol" layer - almost entirely STEERING (prompt-borne).

## Gate decisions (2026-07-23)

1. Reliability = STEERING-ONLY. Trust a steered sub-agent to call `request_input`;
   do NOT add a DONE-in-prose backstop (keep the explicit-signal design the
   original comms spike chose). A model that ignores the steering and stops in
   prose ends DONE and is only visible via the normal agent list; revisit only if
   the live probe shows steering is unreliable.
2. Notification default = POLL + OPT-IN WAKE. `auto_wake` stays OFF by default (no
   unattended orchestrator turns); STEER the orchestrator to poll `pending_agents`
   at end-of-turn, answer each via `message_agent`, and `acknowledge(id)`.
   `auto_wake` remains available (BC4) for fully hands-off operation.

## The clean protocol (decided)

- SUB-AGENT side: a new sub-agent steering preamble - same strip markers as
  `STEERING_PREAMBLE` so `strip_steering` still cleans titles/transcripts - riding
  the turn prompt of a tool-having codex sub-agent: "if you are blocked or need a
  decision/approval you cannot safely make yourself, call `request_input(question)`
  with a clear question and STOP; do not guess - the orchestrator will answer and
  resume you." Gated to codex sub-agents that ACTUALLY have the tool (agent role,
  `agent_tools_enabled`, `agent_id` present) - never a claude sub-agent (no scufris
  MCP) or the orchestrator.
- ORCHESTRATOR side: extend the orchestrator's steering with the comms protocol -
  "at the end of a turn, call `pending_agents` to find sub-agents that need you;
  for each, answer it via `message_agent` (this resumes its session) and then
  `acknowledge(id)` so it stops pending."
- ANSWER delivery is already clean: `message_agent` resumes the sub-agent's session
  with the answer as its next turn, and the `SessionRegistry` keeps the id stable
  across the wait, so the sub-agent continues its own conversation with the answer
  in context.

## Why steering is the whole job (not more mechanism)

The channel is complete and reviewed (BC1-BC4). The only thing missing is that the
models do not know it exists. Per the codex lesson, the fix is prompt-borne
steering on both roles. No new endpoints, tools, or state - two preambles and
their tests. The empirical question ("does the steering actually flip tool
choice?") is answered by a LIVE PROBE (a real backend), exactly as the lesson's
original probe was - so each task carries a `manual:` DoD for that, batched to the
flow Finish.

## Seeded tasks

Seeded as tatr tasks (dependency order): SC1 `20260723-153609` (p38), SC2
`20260723-153615` (p37).

- SC1 - sub-agent `request_input`-when-blocked steering (the missing half of BC2):
  a sub-agent preamble in `_steer`, gated to tool-having codex sub-agents; strip
  markers reused. test: preamble present for a sub-agent turn, ABSENT for the
  orchestrator and for a no-tool turn, and `strip_steering` removes it. manual: a
  real blocked sub-agent calls `request_input`.
- SC2 - orchestrator comms steering: extend the orchestrator steering to poll
  `pending_agents` / answer via `message_agent` / `acknowledge`. test: the
  instruction is present in an orchestrator turn prompt. manual: the orchestrator
  polls and answers a blocked sub-agent end to end.

The mechanism acceptance test remains BC5 (`tasks/20260723-094318`, faked
sub-agent); these two add the steering that makes real models drive it.

## Notes

- Extends the bidirectional-comms spike (`tasks/20260723-001256`) - its missing
  "teach the models" layer.
- Relevant: `codex-tool-choice-only-steers-via-the-turn-prompt`; `_steer` /
  `STEERING_PREAMBLE` / `strip_steering` (`agent.py`, `sessions.py`);
  `_mcp_overrides` agent-role gating (`agent.py`).
