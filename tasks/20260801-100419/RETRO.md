# Retro: Align Telegram and the UI with orchestrator diagnostics

- TASK: 20260801-100419
- BRANCH: fix/telegram-ui-diagnostics-alignment
- REVIEW ROUNDS: 1

## What went well

- Planning re-read the lines the Steps named instead of trusting their prose,
  and caught three stale claims: Steps 2/3 described work `20260801-100415`
  had already done, and `agent-view.ts:148` did not say what the Step quoted.
- The two Telegram tests boot the real app per backend against the real
  `AgentDiagnostics`, so the cross-surface claim is pinned at its own boundary
  rather than re-encoded as a capability table the fix could drift from.
- Round 1 APPROVEd with only MINOR/NIT findings; no rework cycle.

## What went wrong

- The plan sized the change without checking the file-size ratchet.
  `web/src/agent-settings-view.ts` sat at 593 of its 600-line cap, so a
  ~30-line addition forced an unplanned 230-line module split mid-work. That
  seemed sound at plan time because the diff was framed as "delete two
  unwraps" - small by intent - and the cap was never consulted.
- The split then stranded three pointers at the old file
  (`scufris/README.md:356`, `scufris/telegram/text.py:58`,
  `web/src/agent-settings-view.ts:62`), inside the very contract section this
  task added to be the one source those language-local copies point back to.
  Review R1.1.
- A Step's literal text asked for a `DECISION D4 of tasks/<id>` citation in
  `web/src/agent-view.ts`, which AGENTS.md:103 forbids in code comments. The
  plan wrote an instruction that violated a repo rule. Review R1.5.
- No `CHANGELOG.md` bullet, though both sibling tasks in this epic added one
  and this changes operator-visible wording on two surfaces. Review R1.3.

## What to improve next time

- Breadth: 641 insertions over 16 files and two languages was mostly
  inherent - the two unwraps are one defect and the DoD's manual check spans
  both surfaces, so DECISION explicitly refused the split. The avoidable 230
  lines are the forced module move.
- Churn: the plan-time question that would have prevented every MINOR is a
  pre-flight sweep of the standing constraints on each file a Step names -
  line-count ratchet, AGENTS.md comment rules, CHANGELOG duty - written into
  the Step, not discovered when an edit trips a gate.
- Context: no measured pressure, no compaction, no checkpoint. One delegation,
  the round-1 out-of-context reviewer, which paid for itself: five of six
  findings came from it, and the recording pass re-derived two of them
  independently before accepting.

## Action items

- `20260803-042958` carries the six open Round 1 MINOR/NIT findings and the
  `resetsIn` duplication between `web/src/chat-sidebar.ts:96` and
  `web/src/agent-settings-panels.ts:116`.
- Knowledge: a plan Step that adds lines to a file must state that file's
  current size against its ratchet cap, so the split is planned rather than
  discovered.
