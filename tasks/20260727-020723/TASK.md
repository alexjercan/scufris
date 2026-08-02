# Orchestrator logs food from plain language: steer STEERING_PREAMBLE to the den-journal/macros tool chain

- PRIORITY: 55
- TAGS: feature, agents, mcp, journal, codex
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

The orchestrator should log food from plain natural language. Today, saying
"log that I had 2 eggs" or "I ate 2 eggs" does NOT reliably trigger the
food-logging tools; the operator had to spell out the tool chain by hand
("use `macros_lookup` ... then `journal_add_macros`") for it to work. After
this change, a bare "I ate 2 eggs" (and the sibling journal phrasings) makes
the orchestrator reach for the right tools on its own.

Observable done: with the-den + macros DB configured, a turn whose only user
text is "log that I had 2 eggs" (no tool names mentioned) drives
`macros_lookup` then `journal_add_macros` and reports the updated daily total.

## Root cause

The orchestrator (codex) already HAS the tools - `macros_lookup`,
`macros_search`, `macros_add_food`, and `journal_add_macros` are registered and
orchestrator-scoped (mcp_server.py:846-896, kept by `apply_role`). What is
missing is TOOL-CHOICE STEERING.

Per the ledger lesson `codex-tool-choice-only-steers-via-the-turn-prompt`
(LESSONS.md), codex ignores the "soft" instruction channels for tool choice -
tool descriptions, an instructions file, and AGENTS.md all yielded 0 MCP calls
in the live probe; only a preamble prepended to the TURN PROMPT made it prefer
the MCP tools. The orchestrator's turn-prompt steering is `STEERING_PREAMBLE`
(scufris/sessions.py:70), currently two clauses: the host-tools clause
(host_stats/disk_usage/list_processes) and the comms clause. Neither mentions
the den journal or the food-lookup chain, so the food tools live only on the
soft channel codex under-weights - hence the operator had to name them by hand.

## Fix (direction, pending plan gate)

Add a den-journal / food-logging CLAUSE to the SINGLE `STEERING_PREAMBLE`
block. Two constraints from the ledger MUST hold:

- `orchestrator-steering-is-one-block-two-clauses`: it stays ONE
  `[scufris-tools]...[/scufris-tools]` block (strip_steering removes only the
  first leading block). The new guidance is a THIRD clause composed with `\n`
  inside the same block, never a second sentinel block.
- `ground-steering-text-in-the-real-tool-signatures`: name the tools and their
  args verbatim from mcp_server.py. The meal chain is real and exact:
  `macros_lookup("egg 2p")` returns `egg 2pc,12,0,10`, which is precisely the
  CSV row `journal_add_macros(row)` accepts - so lookup-then-log chains with no
  reshaping. If the food is unknown, `macros_search` finds the name and
  `macros_add_food` adds it first.

## Scope decision (resolved at gate 2026-07-27)

RESOLVED: general den-journal clause (operator picked it at the plan gate).
Narrow (food/macros only) vs. general (whole den-journal surface). The same
one-clause mechanism can also point the orchestrator at journal_add_task /
journal_complete_task / journal_toggle_habit / journal_log_weight /
journal_add_note / journal_show for the analogous "add a task", "check off
gym", "log 80kg", "jot this down" phrasings, which share the identical
soft-channel gap. Recommended: general clause (fixes the whole class in one
clause), with the meal chain called out explicitly. Confirm at the gate.

## Steps

- [x] Add a `_JOURNAL_CLAUSE` in scufris/sessions.py and compose it into
      `STEERING_PREAMBLE` as a THIRD clause inside the SAME single
      `[scufris-tools]...[/scufris-tools]` block (host-tools + comms + journal,
      joined with `\n`). The clause: (a) points the orchestrator at the den
      journal tools (journal_show to read; journal_add_task /
      journal_complete_task / journal_toggle_habit / journal_log_weight /
      journal_add_note / journal_add_macros to write) for plain-language
      journal facts instead of memory/file edits; (b) spells out the meal
      chain verbatim: call `macros_lookup(query)` with food+amount (e.g.
      "egg 2p"); its `<food> <amount><unit>,<protein>,<carbs>,<fat>` row is
      exactly what `journal_add_macros(row)` takes, so pass it straight
      through; use `macros_search` / `macros_add_food` first if the food is
      unknown. Every tool name/arg matched verbatim against mcp_server.py.
- [x] Tests (test_agent.py + test_sessions.py):
      - orchestrator `STEERING_PREAMBLE` contains the meal chain
        (`macros_lookup` and `journal_add_macros` both present).
      - it stays ONE block: exactly one `_STEER_OPEN` / one `_STEER_CLOSE`;
        `strip_steering(_steer(..., is_orchestrator=True))` round-trips back to
        the raw prompt (invariant from
        `orchestrator-steering-is-one-block-two-clauses`).
      - the sub-agent preamble (`AGENT_STEERING_PREAMBLE`) does NOT gain the
        journal clause.
- [x] Verify: `ruff check .`, `mypy .`, `python -m pytest` green (the
      `nix flake check` gate).

## Definition of Done

- [x] `STEERING_PREAMBLE` names the `macros_lookup` -> `journal_add_macros`
      chain (and the wider den-journal tools) verbatim, as a single sentinel
      block. (test: test_agent.py / test_sessions.py)
- [x] `strip_steering` still fully cleans an orchestrator-steered prompt - one
      block, count=1 safe. (test: round-trip assertion)
- [x] Full QA gate green. (cmd: `python -m pytest`; `ruff check .`; `mypy .`)
- [ ] A live turn whose only user text is "log that I had 2 eggs" (no tool
      names) drives `macros_lookup` then `journal_add_macros` and reports the
      updated daily total. (manual: operator confirms against the-den +
      macros DB, since it needs a real codex turn - not a CI test)

## Implementation Notes

Added `_JOURNAL_CLAUSE` to scufris/sessions.py and composed it as the third
clause of the single `STEERING_PREAMBLE` block (host-tools + comms + journal,
joined with `\n` inside one `[scufris-tools]...[/scufris-tools]` sentinel).
Nothing else changed - the fix is purely tool-choice steering, since the tools
were already registered and orchestrator-scoped.

Why steering and not tool descriptions: the ledger lesson
`codex-tool-choice-only-steers-via-the-turn-prompt` proved codex ignores the
soft channels (docstrings, instructions file, AGENTS.md) for tool choice; only
a turn-prompt preamble moves it. So the docstrings on `macros_lookup` /
`journal_add_macros` (which already say "PREFERRED" / "Use this for log a
meal") were never going to be enough on their own - hence the operator had to
name the tools by hand.

Ledger constraints honored: kept it ONE block
(`orchestrator-steering-is-one-block-two-clauses`, so `strip_steering`'s
`count=1` still fully cleans it - asserted by a round-trip test); and copied
every tool name/arg verbatim from mcp_server.py
(`ground-steering-text-in-the-real-tool-signatures`). The meal chain is exact:
`macros_lookup("egg 2p")` -> `"egg 2pc,12,0,10"` is byte-for-byte the row
`journal_add_macros(row)` accepts, so no reshaping is needed between the two
calls.

Tests: `test_steer_orchestrator_gets_journal_food_chain` (meal chain + wider
den surface present on the orchestrator, absent on a sub-agent, still
strippable) and `test_orchestrator_steering_stays_a_single_block` (one sentinel
pair) in tests/test_agent.py. Full gate green: ruff, ruff format, mypy (54
files), pytest (all).

The live E2E (DoD #4) is left for the operator: it needs a real codex turn and
WRITES to the operator's the-den journal + macros DB, so it is manual
acceptance, not something to run unprompted from the flow.
