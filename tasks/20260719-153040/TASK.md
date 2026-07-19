# Spike: LLM agent harness for GPT-5.5 via Pro/Plus subscription

- STATUS: CLOSED
- PRIORITY: 0
- TAGS: spike, backlog, agent, llm

## Question

Which LLM agent harness should Scufris embed to power the chat agent, given the
hard constraints: it must drive an external provider's model (target GPT-5.5),
authenticate via a **Pro/Plus subscription rather than a metered API key**, and
integrate cleanly from Python? Evaluate candidates like opencode, OpenAI Codex
(CLI/SDK), and any comparable harness, and recommend one.

## Context

The agent is the third Scufris pillar. Scufris is a Python app (FastAPI). The
subscription-not-API-key constraint is the crux: it rules out the plain
metered API path and points at harnesses that reuse an existing
ChatGPT/Codex-style subscription session. The harness also needs to expose
tool/function calling so the agent can read host metrics and trigger the same
CLI actions the dashboard exposes.

## What a good answer looks like

A recommended harness with the runner-up weighed on: subscription-auth support
(does it actually work without an API key, and is that within terms of use?),
Python integration story (native SDK vs shelling out to a CLI vs a local
server/daemon it exposes), tool-calling support, model/provider flexibility
(can it target GPT-5.5), and maintenance/packaging cost under nix. Flag any
legal/ToS or reliability risk explicitly - "not viable because ..." is a valid
and useful outcome.

## Candidate directions to explore (diverge before converging)

- **opencode** - how it authenticates, whether it can use a subscription, and
  how Python would drive it (CLI, server mode, or library).
- **OpenAI Codex (CLI / SDK)** - subscription vs API-key auth, and the Python
  integration surface.
- **A thin custom harness** over whatever subscription-session transport is
  available - most control, most maintenance, most ToS exposure.
- **Concede the constraint** - fall back to an API key (or a local model) if no
  harness supports subscription auth safely; document what that costs.

## Notes

- Output per the /spike skill: write `tasks/<id>/SPIKE.md`, seed direction-level
  tasks, close this spike task.
- Keep the provider/harness behind one interface in Scufris so the choice is
  reversible if a provider changes its terms.
- Verify auth claims against current provider docs/terms at spike time; do not
  rely on memory. Note the knowledge-cutoff risk for GPT-5.5 specifics.
