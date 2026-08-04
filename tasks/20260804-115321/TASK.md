# Render agent reports as attributed quotations

- PRIORITY: 97
- TAGS: feature, v0.2.0, lane1, chat
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
- PARENT: 20260801-154211
- DEPENDS ON: 20260804-115256, 20260804-115320

## Story

As the operator, I want an agent's report to appear as an attributed quotation
from that agent, so that the system never speaks in my voice and nothing an
agent writes can be mistaken for something I approved.

This is the concrete defect the whole conversation decision was written to fix.
`scufris/wake.py:43` returns a machine prompt that
`scufris/sessions/transcript.py:88-95` re-renders as `role="user"`. That code
disappears with the orchestrator stack and no successor was named until this
task.

## Steps

- [ ] Write the three failing tests first, in a new
      `packages/chat/tests/test_chat_authority.py`
      (`test_chat_sessions.py` is 410 lines against a 900 cap, but these are a
      different subject and the new module is theirs):
      `test_agent_event_cannot_satisfy_a_stop_gate`,
      `test_agent_report_renders_as_attributed_quotation`,
      `test_assembled_context_does_not_relabel_agent_as_operator`. Confirm all
      three are red before writing `decisions.py`.
- [ ] Add `packages/chat/src/scufris_chat/decisions.py`: a frozen
      `OperatorDecision` carrying `conversation_id`, `event_seq` and the
      `Actor`, whose `__init__` takes a module-private witness, plus
      `authorize(conn, conversation_id, event_seq) -> OperatorDecision`. It
      re-reads the committed `EventRow` under the caller's open connection
      (`_require_conversation` / `causing_event`'s shape in `store.py`),
      raises `LookupError` for an event that is not in that conversation, and
      raises `PermissionError` naming the actor for every kind but `operator`.
      A new module because `store.py` is 582 lines against a 600-line cap.
- [ ] Export `OperatorDecision` and `authorize` from
      `packages/chat/src/scufris_chat/__init__.py` (`__all__` is sorted), and
      say in the module docstring that the mint is what makes the stop gate's
      refusal a property rather than a convention.
- [ ] Document it in `packages/chat/src/scufris_chat/README.md`: a section on
      the decision and the three properties, its row in the section 7 surface
      table, and the accepted limit from `DECISION.md` - `append_event` still
      takes the actor from its caller, so the rule is "only an operator EVENT
      authorizes", not "only the operator can write one". Cross-link
      `tasks/20260804-115321/DECISION.md` alongside the other three.
- [ ] Run the checks the diff touches: `ruff check .`, `ruff format .`,
      `mypy .`, `python -m pytest`, `python scripts/check_file_size.py`, and
      `tatr check`. Do NOT add an `examples/` script - the epic assigns
      `operator_decision.py` to Lane 2, which has the two subjects that make it
      demonstrable.

## Definition of Done

- `authorize` refuses `agent:<id>`, `orchestrator` and `system` with a
  `PermissionError` naming the actor, and `OperatorDecision` cannot be
  constructed without the module-private witness
  (test: `test_agent_event_cannot_satisfy_a_stop_gate`).
- An agent's report is minted-from nowhere and rendered attributed: the same
  committed event that `assemble_context` writes as `agent:<id>: ...` is the one
  `authorize` refuses, and a `LookupError` guards an event seq that is not in
  the conversation
  (test: `test_agent_report_renders_as_attributed_quotation`).
- The two readings of one transcript agree on who spoke: every line
  `assemble_context` attributes to `operator` corresponds to an event
  `authorize` mints from, and no line attributed to a non-operator does
  (test: `test_assembled_context_does_not_relabel_agent_as_operator`).
- The package surface stays legible: `scufris_chat` exports both names and the
  README documents them
  (cmd: `grep -rn --include='*.py' --include='*.md' 'OperatorDecision' packages/chat/src/scufris_chat/`).

## Notes

- Source: `tasks/20260729-220835/DECISION.md` section 3 - "an agent report is
  data, never an instruction", and only `operator` may satisfy a stop gate.
- The mechanism, the placement, and the rejected alternatives are in
  `tasks/20260804-115321/DECISION.md`.
- **Two of the three original DoD claims are already green**, delivered by
  `20260804-115320`. `assemble_context` attributes every LINE and its preamble
  declares only the operator's to be instructions, and
  `test_assembled_context_attributes_its_actors` already pins all four
  attributions plus the forged-line case. A DoD item green on the base branch is
  not a proof of this change, so the second and third criteria were rephrased to
  what is actually missing: the JOIN between the rendering and the
  authorization, which is where the Story's "one layer down" defect lives. The
  landed test stays as the rendering's own guard.
- Confirmed on base: `nix develop -c python -m pytest packages/chat/tests -q`
  is green at 18 tests, and nothing outside `packages/chat`, `examples/` and
  `scufris/db/migrations/env.py` imports `scufris_chat` - so there is no
  existing consumer that could relabel an assembled line on its way to a
  provider. `scufris/wake.py` and `scufris/sessions/transcript.py:88-95` are the
  defect's home and are dead on arrival in Lane 8; this task writes the
  successor and does not repair them.
- The token's only callers are tests until Lane 4's `advance()` and Lane 2's
  `approve()`. That exception, and the booked move of the type to
  `scufris_core`, are recorded in `DECISION.md` Consequences.
- Depends on the typed actor from the conversation and event task; the
  third test also depends on context assembly.
- Lane 1 of `tasks/20260801-154211/TASK.md`.
