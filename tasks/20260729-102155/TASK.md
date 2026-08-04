# Add repository flow lifecycle and scheduling conformance

- PRIORITY: 0
- TAGS: chore, backlog, flow, tatr
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Story

As a `$flow` user, I want repository conformance to reject lifecycle states
that contradict the documented methodology, so that task records reliably
prove planning, review, verification, retro, scheduling, and landing status.

## Steps

- [ ] Write failing fixture tests for missing scheduling tags, multiple
      scheduling tags, CLOSED tasks not at `FLOW STEP: DONE`, missing approved
      plans on flow-managed work, and claimed proofs without recorded results.
- [ ] Add a project-owned conformance command that composes `tatr check` with
      the Scufris-specific scheduling and flow rules.
- [ ] Define how squash landing is verified without treating harmless historical
      commits as current failures.
- [ ] Repair current task metadata that violates the new rules, preserving
      append-only history and recording any exception explicitly.
- [ ] Add the conformance command to the canonical QA gate and contributor
      instructions.
- [ ] Document the boundary between generic tatr rules and Scufris policy so
      reusable rules can later be contributed to tatr itself.

## Definition of Done

- Fixtures for every invalid lifecycle/scheduling state fail with actionable
  task IDs and rule names (test: `test_flow_record_conformance`).
- Every new task must have exactly one `backlog` or release scheduling tag
  (test: `test_task_requires_one_scheduling_tag`).
- A CLOSED flow-managed task cannot remain UNDERSTANDING through COMPOUNDING
  (test: `test_closed_flow_task_must_be_done`).
- Repository task and lesson checks pass
  (cmd: `tatr check --ledger LESSONS.md && ./scripts/check-flow-records`).

## Notes

- Epic: 20260729-102149.
- Start with a project-owned check because the tatr source is not part of this
  repository. Promote generally useful rules upstream in a separate tatr task.
- Do not rewrite substantive historical records solely to make metrics look
  cleaner; use explicit lifecycle annotations where appropriate.
