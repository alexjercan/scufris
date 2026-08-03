# Retro: Spike: define the actor-aware orchestrator conversation and flow-control model

- TASK: 20260729-220835
- BRANCH: master
- REVIEW ROUNDS: 1

## What went well

- Locating the defects in the SHIPPING code before comparing options. The wake
  prompt rendering in the operator's own voice (`scufris/wake.py:44`,
  `scufris/sessions/transcript.py:94`) and the in-memory `_announced` dedupe
  (`scufris/telegram/approvals.py:78`) are what turned "we should have actors"
  from a preference into a defect with a structural fix. Options argued against
  named line numbers beat options argued against principles.
- The mockup carried the acceptance round by itself. One self-contained static
  file, fixture data, a step index, three linked views - and the round approved
  the model on the first pass with no redirection. The information hierarchy
  was arguable BECAUSE it was clickable; a prose description of the same model
  would have produced a discussion about wording.
- Stating the mockup's limitations while it was in front of the reviewer -
  fixtures, no SSE, no restart, one path per branch - kept the round about the
  model instead of about the demo's gaps.
- Re-checking rather than re-litigating the prior rejections. The full-transcript
  option was declined in `tasks/20260724-111839/SPIKE.md`; this spike confirmed
  nothing had changed and moved on, instead of re-deriving it.
- Partial supersession as a tool. Rather than quietly contradicting
  `tasks/20260720-184150/SPIKE.md`'s "orchestration pipelines are dropped", the
  decision named exactly what still stands (a generic engine with no authority)
  and what is superseded (a coordinator over a state machine tatr owns). The
  distinction is the authority, and writing it down is what stops the next
  reader from re-opening it.

## What went wrong

- The spike was scheduled against v0.3.0, behind a polish release the
  maintainer did not want. Its acceptance round became a release re-cut: 21
  tasks demoted to backlog, the intervening release cancelled, this model moved
  to v0.2.0. The signal was available earlier - the "entry criteria" list in
  `tasks/20260801-154211/TASK.md` had grown to seven prerequisite items for a
  product claim none of them made - and nobody read that list as the symptom it
  was.
- The DECISION record shipped saying `STATUS: ACCEPTED` while it was in fact
  proposed, with a paragraph in the body explaining that the real gate was
  elsewhere. That works, and it is a workaround for `tatr` having no PROPOSED
  status, but a status field that has to be corrected by prose is a status field
  the reader can misread at a glance.
- The decision's "Paid" section budgeted for migrating existing data under a
  retention policy that does not exist. The acceptance round deleted that cost
  outright by dropping the database. Reasonable at the time, but it shows the
  spike assumed continuity nobody had asked for - the cheaper answer was
  available and unexamined.

## What to improve next time

- When a task's "entry criteria" list outgrows its own deliverable, stop and ask
  whether the release is real. Seven prerequisites for one product claim is a
  scheduling smell, not a thorough plan.
- Ask "what if we keep nothing?" as a standing question in any spike that
  proposes a new store. It costs one paragraph and would have been the right
  answer here.
- Build the clickable artifact earlier. It settled in one round what the prose
  had been circling for several, and it is cheap - fixtures and a step index, no
  build and no server.
- Keep naming line numbers in spikes. Every load-bearing claim in this one
  points at code, and that is why the review round argued about the model rather
  than about whether the problems were real.

## Action items

- [x] Record the release re-cut and the no-backwards-compatibility rewrite in
      the DECISION's ratification paragraph.
- [x] Promote epic 20260729-102157 to lead v0.2.0, restated as an operating
      surface, and demote the polish and observability children.
- [ ] Route the `uv` workspace packaging question raised at acceptance to its
      own record; it is not a conversation-ownership question and does not
      belong in this decision.
- [ ] Carry the spike's open questions - retention, summary versioning, event
      granularity, re-seed eagerness, the `SCUFRIS_ORCH_SESSION_ID` rename, and
      where the guard service lives - into the implementation tasks that hit
      them, rather than into a second architecture round.
