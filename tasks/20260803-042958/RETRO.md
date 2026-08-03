# Retro: Clear the round-1 MINOR findings from the diagnostics alignment

- TASK: 20260803-042958
- BRANCH: chore/diagnostics-minors
- REVIEW ROUNDS: 2

## What went well

- The one behavioral item in a nine-step cleanup batch, R1.2, was driven RED
  FIRST and the red was reproduced on the base branch before the fix. The plan
  had already identified it as the only item with a user-visible effect and the
  only one needing a test, and that call held.
- Plan-time pointer re-derivation paid for itself. Three of the six inherited
  findings carried drifted line numbers, and one, R1.6, was 2/3 right -
  dropping `statusPanel`'s export would have broken the build. Checking each
  pointer against the code before writing Steps turned a would-be BLOCKER into
  a documented carve-out.
- Round-2 fixes were each confirmed against the code before being accepted,
  rather than applied on the finding's say-so.

## What went wrong

- R1.1 (MAJOR): the CHANGELOG bullet this task added named Telegram surfaces
  that do not exist - a `memory` card with no command, and `health`, whose
  renderer emits no capability string. A task whose whole Story is "make the
  pointers match the code" shipped a new pointer that did not. The root cause
  is the shape of the proof: DoD proof 3 grepped that a bullet mentioning the
  wording EXISTS. Presence was provable by grep; accuracy was not, and nothing
  else checked it.
- R1.2 (MINOR): an 89-char line against the repo's 88 reached review because
  `nix flake check` runs `ruff check` but not `ruff format --check`. The
  branch's only formatting regression slipped past the gate the task trusted.
- R2.1 (NIT): the round-2 close-out described a divergence from R1.3's proposal
  that never happened - the delivered wrap was character-identical to what the
  finding proposed. Corrected in the same phase.

## What to improve next time

- Breadth: the diff is large in step count, not in surface - nine items, all
  from one inherited review batch, all in comments, docs and one renderer. The
  `resetsIn` dedupe was the only addition, kept in scope with a stated reason
  (it touches the same export list R1.6 does, so splitting it would race two
  commits over one file). No missed independently-landable split.
- Churn: the from-scratch challenge on the DoD, not on the design, is what
  would have prevented round 1. Both round-1 substantive findings are proof
  gaps rather than code flaws: a grep pins that a claim was written, never that
  it is true, and the local check suite is a superset of the CI gate. When a
  Step's deliverable is a prose CLAIM about which surfaces do X, the DoD owes a
  proof that enumerates those surfaces from the code - here, `_quota_reading`'s
  call sites - not a grep for the sentence.
- The DoD's greps are line-scoped, so a comment rewrap can break a proof
  without changing meaning. A proof that means "these two tokens appear
  together" should say so rather than relying on line adjacency.
- Context: reviews ran out-of-context in both rounds as the protocol requires;
  no compaction, checkpoint or handoff was observed.

## Action items

- 20260803-182446 (seeded during round 2): add `ruff format --check` to the
  `nix flake check` gate so local and CI formatting agree. Pre-existing and a
  flake change, so deliberately not fixed here.
- R2.1 stays open as a NIT with its correction applied to TASK.md; the box is
  unticked because the round's reviewer is out-of-context.

## Landing message

```
chore(diagnostics): align the three-state pointers with the code

Clear the nine round-1 cleanup items from the diagnostics alignment: fix
`render_settings_summary` to fall back to `usage.secondary` so it agrees with
`/settings usage` on a secondary-only quota, collapse `render_usage`'s window
comprehension and its dead second guard, correct the `capabilityText` module
pointer in `scufris/README.md`, `scufris/telegram/text.py` and
`web/src/agent-settings-view.ts`, drop a task ID from a web comment and two
unused exports, dedupe `resetsIn` into `web/src/common.ts`, and add a
CHANGELOG bullet for the three-state operator wording.
```
