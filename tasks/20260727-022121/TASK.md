# Auto-delegate task implementation to a backend sub-agent from plain language (orchestrator + sub-agent steering)

- PRIORITY: 60
- TAGS: feature, agents, mcp, codex, steering
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

The operator should delegate task implementation to a backend agent in plain
language. Today, "implement task X using codex/claude" does not reliably make
the orchestrator create-and-run the right agent, and when it does, the spawned
sub-agent does nothing useful (the reported run: 1 turn, 0 tool calls, 76534
in / 6320 out, then "done" - it narrated instead of acting).

After this change:
- "implement task X using codex" / "have claude do task Y" makes the
  orchestrator find the project, create (or reuse) an agent on the named
  backend with a write-capable permission mode, and run it with the task as its
  goal - without the operator spelling out create_agent/run_agent.
- The spawned sub-agent knows its job is to carry the assigned task to
  completion end-to-end (understand -> implement -> run the project's checks),
  and to call request_input(question) and STOP when genuinely blocked - rather
  than replying with a plan and finishing.

## Root cause

Same class as 20260727-020723 (food-logging steer): the ORCHESTRATOR already
has the delegation tools - list_projects, list_agents,
create_agent(name, project_id, backend, permission_mode),
run_agent(agent_id, goal), agent_status(agent_id) - but nothing on the TURN
PROMPT steers it to chain them for "implement task X using codex". codex only
honors tool-choice steering on the turn prompt
(codex-tool-choice-only-steers-via-the-turn-prompt), so the tool docstrings
alone do not trigger the create-then-run chain.

On the sub-agent side, AGENT_STEERING_PREAMBLE (scufris/sessions.py) currently
carries ONLY the request_input-when-blocked instruction. It never tells the
agent to actually implement the task. The failed run was told (in its goal) to
"use the flow skill", but a codex sub-agent has no Claude Code skills to load,
so it produced framing text and stopped with 0 tool calls. The fix is a
backend-agnostic work clause that gives actionable steps on the turn prompt,
not a pointer to a skill only claude can load.

## Decision (see DECISION.md)

Sub-agent steering is BACKEND-AGNOSTIC: it instructs every sub-agent to
implement its assigned task end-to-end and signal via request_input, without a
hard dependency on the flow skill (codex cannot load it). It MAY mention "use
the flow skill if available" for the claude backend, but the actionable steps
carry the instruction on their own. Scope this cycle is STEERING ONLY (prompt
changes, harness-tested); if the live "did nothing" failure persists after
this, it becomes a separate follow-up task.

## Steps

- [x] Add `_DELEGATION_CLAUSE` in scufris/sessions.py and compose it into
      `STEERING_PREAMBLE` as a FOURTH clause inside the SAME single
      `[scufris-tools]...[/scufris-tools]` block (host-tools + comms + journal +
      delegation, joined with `\n`). The clause steers the orchestrator: when
      the operator asks to implement / work / delegate a task to a codex or
      claude agent ("implement task X using codex", "have claude do task Y"),
      (a) find the project with `list_projects`; (b) reuse a fitting agent from
      `list_agents` or `create_agent(name, project_id, backend, permission_mode)`
      with `backend` = the named provider and a WRITE-capable
      `permission_mode` (`edit` or `auto` - the default `manual` is read-only,
      so an implementing agent must not use it); (c) `run_agent(agent_id, goal)`
      with the task id/path and what to do as the goal; (d) follow with
      `agent_status(agent_id)` and answer its `request_input` signals via the
      existing comms clause. Every tool name/arg matched verbatim against
      mcp_server.py.
- [x] Extend `AGENT_STEERING_PREAMBLE` (scufris/sessions.py) with a
      backend-agnostic WORK clause, kept inside its single sentinel block
      alongside the existing request_input instruction: the sub-agent's job is
      to carry the assigned task/goal to completion end-to-end - understand it,
      make the changes, run the project's checks - and NOT to stop after
      describing a plan; if a flow skill is available (claude backend) it may
      run it, but the actionable steps stand alone so codex works too; when
      genuinely blocked or needing a decision it cannot safely make, call
      `request_input(question)` and STOP.
- [x] Tests (tests/test_agent.py):
      - orchestrator `STEERING_PREAMBLE` contains the delegation chain
        (`create_agent` and `run_agent` present); a sub-agent turn does NOT.
      - the sub-agent `AGENT_STEERING_PREAMBLE` tells it to implement the task
        end-to-end AND still to call `request_input`; the orchestrator's
        delegation tools are NOT in the sub-agent preamble.
      - both preambles stay ONE block (exactly one `[scufris-tools]` /
        `[/scufris-tools]` each); `strip_steering` round-trips a steered prompt
        of each role back to the raw text.
- [x] Verify: `ruff check .`, `mypy .`, `python -m pytest` green.

## Definition of Done

- [x] `STEERING_PREAMBLE` names the delegation chain
      (`list_projects` -> `create_agent`/`list_agents` -> `run_agent` ->
      `agent_status`) verbatim, steers to a write-capable permission mode, and
      stays a single sentinel block. (test: tests/test_agent.py)
- [x] `AGENT_STEERING_PREAMBLE` tells the sub-agent to implement its task
      end-to-end and still to `request_input` when blocked, backend-agnostic,
      single block. (test: tests/test_agent.py)
- [x] `strip_steering` still fully cleans a steered prompt for BOTH roles -
      one block each, count=1 safe. (test: round-trip assertions)
- [x] Full QA gate green. (cmd: `python -m pytest`; `ruff check .`; `mypy .`)
- [ ] A live "implement task X using codex" turn (no tool names) makes the
      orchestrator create + run a write-capable agent that actually works the
      task and reports/asks via request_input rather than finishing at 0 tool
      calls. (manual: operator confirms against a real delegated run - needs
      live codex/claude backends, not a CI test)

## Implementation Notes

Two edits, both in scufris/sessions.py; no behavior wiring changed, only the
turn-prompt steering strings (parallels 20260727-020723).

- `_DELEGATION_CLAUSE` added as the fourth clause of the single
  `STEERING_PREAMBLE` block. It names the real orchestrator tools verbatim
  (list_projects / list_agents / create_agent / run_agent / agent_status) and
  explicitly steers to a WRITE-capable permission_mode - this is the one line
  that ties back to the reported failure: create_agent defaults to "manual"
  (read-only), which alone yields "0 tool calls" no matter how good the goal.
- `AGENT_STEERING_PREAMBLE` gained a work clause before the existing
  request_input sentence, still inside its one sentinel block. Per DECISION.md
  it is backend-agnostic: concrete turn-prompt steps (carry the task to
  completion, run the checks, don't stop at a plan) that work on codex, with
  the flow skill mentioned only as an optional aid for claude - because the
  failed run leaned on a flow skill codex can't load and stalled.

Both preambles stay ONE `[scufris-tools]` block (asserted by
`test_orchestrator_steering_stays_a_single_block` and
`test_agent_steering_stays_a_single_block`), so `strip_steering`'s count=1
still cleans titles/transcripts for both roles.

Tests added in tests/test_agent.py:
`test_steer_orchestrator_gets_agent_delegation_chain`,
`test_steer_agent_told_to_implement_the_task_end_to_end`,
`test_agent_steering_stays_a_single_block`. Full gate green (ruff, ruff
format, mypy 54 files, pytest).

DoD #4 (live delegated run) is manual: it spawns real background codex/claude
processes and writes to a project, so it is operator acceptance, not a CI test.
The steering fix is expected to address the reported root cause (skill-only
sub-agent goal + no work steering), but confirming that needs a live run.
