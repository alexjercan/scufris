# Reconcile agent_enabled default vs README off-by-default docs drift

- PRIORITY: 42
- TAGS: docs, agent
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As a reader following the README, I want its statements about whether agents run
by default to match the actual code default, so I do not export a flag that is
already the default or assume agents are off when they are on.

## Context (grounded)

`scufris/config.py:95` sets `agent_enabled: bool = True` - agents are ON by
default. But `README.md:52` says "Agents are **off by default** and provisioned by
the operator", and the quickstart at `README.md:92` tells the reader to
`export SCUFRIS_AGENT_ENABLED=1` (redundant if already true). `.env.example:37`
also carries `SCUFRIS_AGENT_ENABLED=1`. Discovered during the comms-arc review.

DECISION NEEDED (confirmed with the user at the flow gate): the user asked to "fix
the DOCS", i.e. keep `agent_enabled=True` and correct the README/quickstart to
match. (The alternative - flip the default to `False` to match the README's
"operator-provisioned opt-in" framing - was NOT chosen.) If the gate answer
differs, this task flips to a one-line config change instead.

## Steps

- [x] Correct `README.md:52` so it no longer says agents are "off by default"
      (state the true default: agents are enabled by default; the operator still
      provisions/authenticates a backend before they do anything).
- [x] Fix the quickstart (`README.md:92`) so it does not imply `agent_enabled` must
      be exported to turn agents on (note the flag exists to DISABLE, i.e.
      `SCUFRIS_AGENT_ENABLED=0`, or drop the redundant export). Keep the auth step.
- [x] Sweep the rest of README + `.env.example` for any other statement that
      contradicts `agent_enabled=True` (e.g. the `.env.example:37` comment framing).
- [x] Grep the doc surface for other "off by default" agent claims and fix any that
      are wrong (exclude `tasks/`).

## Definition of Done

- No README/`.env.example` statement contradicts `agent_enabled=True`.
  (cmd: `grep -n "off by default" README.md` returns nothing about agents being off)
- The quickstart no longer implies you must enable agents to use them.
  (manual: read README Agents section)
- `ruff check .`, `mypy`, `python -m pytest` still green (docs-only, but verify no
  doctest/link check breaks). (cmd: `python -m pytest`)

## Notes

- Docs-only unless the gate flips the decision to a config default change.
- Surfaced by the bidirectional-comms arc review (umbrella 20260723-192825).
