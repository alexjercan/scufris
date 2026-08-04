# Spike: teach both sides the comms protocol (steer request_input-when-blocked + orchestrator poll/answer)

- PRIORITY: 39
- TAGS: spike, agents, backend
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story (spike)

BC1-BC4 built the whole message CHANNEL (request_input -> WAITING outcome -> wake
bridge / pending_agents poll -> answer via message_agent -> resume), but neither
model is taught the PROTOCOL: a sub-agent has the request_input tool yet is never
steered to use it when blocked, and the orchestrator is not steered to poll,
answer, or acknowledge. This spike decides how both sides learn the protocol so
they pass messages cleanly.

## Spike output

Full research, the two gate decisions (steering-only reliability; poll + opt-in
wake), the clean protocol, and the seeded tasks are in `SPIKE.md` next to this
file. Summary: the fix is prompt-borne STEERING on both roles (per
`codex-tool-choice-only-steers-via-the-turn-prompt`), no new mechanism.

Seeded tasks:

- SC1 - sub-agent request_input-when-blocked steering (the missing half of BC2).
- SC2 - orchestrator comms steering (poll pending_agents / answer / acknowledge).

Each carries a `manual:` live-probe DoD ("does the steering actually make a real
model use the protocol", empirical per the lesson). The mechanism acceptance test
stays BC5 (`tasks/20260723-094318`).

## Notes

- Extends the bidirectional-comms spike (`tasks/20260723-001256`) - its missing
  "teach the models" layer.

## Spike output (close record, 2026-07-23)

`SPIKE.md` written (RECOMMENDED). Gate decisions: (1) steering-only reliability -
trust a steered sub-agent to call request_input, no DONE-in-prose backstop; (2)
poll + opt-in wake - auto_wake stays OFF, steer the orchestrator to poll
pending_agents / answer via message_agent / acknowledge. The fix is prompt-borne
STEERING on both roles (no new mechanism), per
`codex-tool-choice-only-steers-via-the-turn-prompt`.

Seeded 2 tatr tasks: SC1 `20260723-153609` (sub-agent request_input steering), SC2
`20260723-153615` (orchestrator comms steering). Each carries a manual live-probe
DoD. Mechanism acceptance stays BC5 (`tasks/20260723-094318`).
