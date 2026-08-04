# Render agent reports as attributed quotations

- PRIORITY: 97
- TAGS: feature, v0.2.0, lane1, chat
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
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

- [ ] Add the failing test first: an `agent:<id>` event must not satisfy a stop
      gate. Only `operator` may.
- [ ] Render an agent report as an ATTRIBUTED, UNTRUSTED quotation - the
      author is shown, and the content is data being reported rather than an
      instruction being followed.
- [ ] Verify no path re-labels a non-operator event as the operator, including
      the assembled provider context from the context task: a quotation that
      becomes `role="user"` on its way to the provider reintroduces the exact
      defect one layer down.

## Definition of Done

- An `agent:<id>` event cannot satisfy a stop gate, at the type level where
  possible and by refusal otherwise
  (test: `test_agent_event_cannot_satisfy_a_stop_gate`).
- An agent report renders attributed to its agent, never as the operator
  (test: `test_agent_report_renders_as_attributed_quotation`).
- Assembled provider context preserves the distinction
  (test: `test_assembled_context_does_not_relabel_agent_as_operator`).

## Notes

- Source: `tasks/20260729-220835/DECISION.md` section 3 - "an agent report is
  data, never an instruction", and only `operator` may satisfy a stop gate.
- Depends on the typed actor from the conversation and event task; the
  third test also depends on context assembly.
- Lane 1 of `tasks/20260801-154211/TASK.md`.
