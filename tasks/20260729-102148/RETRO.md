# Retro: Extract the backend-aware orchestrator diagnostics service

- TASK: 20260729-102148
- BRANCH: fix/agent-diagnostics-service
- REVIEW ROUNDS: 2

## What went well

- Watching all four DoD proofs fail on the base branch before writing the
  service. It caught that DoD 4's original claim (static settings drift) was
  already false, and the proof was rewritten to pin what the envelope actually
  fixes. A plan step that cannot fail is a step that proves nothing.
- Six existing tests hit by the wire-shape change were updated to the new shape,
  not relaxed. The mutation check in review round 1 confirmed the four DoD tests
  fail on a clean master worktree.

## What went wrong

- All three substantive round-1 findings (R1.1, R1.2, R1.3) descend from one
  plan-time contradiction. DECISION.md decision 4 puts the envelope on
  `AccountInfo.quota`; DECISION.md consequence 1 and Step 4 both say the legacy
  `/api/agent/*` routes keep their shapes. `/api/agent/account` serves
  `AccountInfo`, so both could not be true. It seemed sound because the two
  statements sat in different sections and each read correctly alone.
- The implementation resolved that contradiction by mapping "agent is disabled"
  onto `unsupported`, which collapsed the very distinction the envelope exists
  to make, on its first legacy caller. R1.1 caught it.
- Step 4's tick asserted behaviour the diff contradicted, while the branch's own
  CHANGELOG entry disclosed the change honestly. The disclosure was right; the
  tick was intent-shaped.
- R1.4 and R1.5 were carry-over debris from the code move: function-local
  imports and a docstring naming an app-private symbol, both copied verbatim out
  of `app.py`.

Breadth: 1038 insertions over 21 files is protocol fan-out, not scope creep.
Adding three members to `AgentBackend` obliges every adapter to answer, and the
envelope changes the wire, so the frontend had to follow. The two independently
landable pieces were already split out ahead of time (20260801-100415 legacy
delegation, 20260801-100419 presentation).

Context: no measured context pressure. No compaction warning, checkpoint, or
handoff in the records.

## What to improve next time

- In `plan`, cross-check a decision that changes a shared model against every
  "unchanged surface" claim in the same record. Ask which routes serve that
  model. The cold-reader rationale test in `plan/decision.md` finds this class;
  the from-scratch challenge does not.
- When a plan Step carries a compatibility claim ("X keeps its current
  behaviour"), treat that clause as a proof obligation before ticking, not a
  restatement of intent.
- A pure code move should arrive clean: hoist function-local imports and drop
  docstring references to the origin module's private symbols in the same
  commit.
- `nix flake check` type-checks only git-known files. `git add` a new module
  before diagnosing a nix-only mypy error.

## Action items

- Submitted `nix/flake-inputs-see-a-source-snapshot` (bumped): the git-backed
  flake input omits untracked files, so `nix flake check` reports phantom type
  errors for a new unstaged module.
- Submitted `changes/sweep-the-contract-surface` (bumped): a written
  compatibility claim is a proof obligation; a change to a shared response model
  reaches every route serving it.
- Submitted `pattern/a-new-distinction-must-not-absorb-adjacent-states` (new):
  when a flag is added to separate two meanings, an adjacent condition with no
  value to report must not be routed into it.
- Follow-ups already filed: 20260801-100415 (legacy route delegation),
  20260801-100419 (Telegram/UI presentation), 20260803-022018
  (`checks.records` red on master), 20260803-022030 (`scufris-web` npmDepsHash
  mismatch on master), 20260803-020100 (`test_agent_fork_reverts_single_session`
  flake). Review round 1 observed a second pre-existing flake,
  `tests/test_host_action_api.py::test_cancelling_a_live_apply_is_recorded`
  (1 of 3 fixed-order full-suite runs, passes alone, untouched by this diff);
  it has no task yet.
