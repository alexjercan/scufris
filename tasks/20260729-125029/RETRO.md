# Retro: Add the host action framework with preview approval and audit

- TASK: 20260729-125029
- BRANCH: feat/host-action-framework
- REVIEW ROUNDS: 3

## What went well

- The privileged helper boundary was the right shape. Keeping argv construction,
  proposal state, apply, and audit inside `scufris-hostd` made the later fixes
  local: the app can still only ask for typed proposals and approve helper-owned
  ids.
- The out-of-context review paid for itself. Round 1 found actual unauthorized
  execution paths, and Round 2 found the two places the first fixes were too
  narrow. The final branch is materially stronger because review re-derived
  claims from source and tests, not from the response prose.
- The task notes captured the failures close to the implementation. `NOTES.md`
  now records the command/preview mismatch, cancellation behavior, secret carrier
  mistake, and audit-flood shape in enough detail for the next host task.

## What went wrong

- R1.1 happened because the approval middleware was phrased as "reject a machine
  credential" instead of "require a real operator identity." That left the
  no-credential path open when auth was off on loopback.
- R1.3 and R2.1 happened because I treated "strip secrets from the agent" as a
  Codex runner fix, not as a property of every model-driven subprocess. The
  hostd secret arrived through the environment, and Claude inherited it until
  the strip moved behind a shared helper with structural coverage.
- R1.6 and R2.2 happened because caller-supplied attribution was allowed to
  influence server decisions. Audit identity and rate-limit identity must come
  from the credential; the request body can only supply descriptive metadata.
- R2.5 happened because a ticked task step mixed framework substrate with future
  UI behavior. The framework records `risk` and `reversal.possible`, but stronger
  confirmation belongs to the approval surfaces, so it needed a receiving task
  requirement rather than a false completed claim.

## What to improve next time

- Write authorization gates as a required principal plus explicit allowed
  principals, and test the absence-of-credential path separately from the
  wrong-credential path.
- For any secret that can enter `os.environ`, enumerate the subprocesses that
  receive the environment and test the recipient's final env. A named secret set
  is not enough unless every spawn is forced through it.
- Keep decision identity and display attribution separate in API models. Fields
  supplied by the caller can label a request, but they must not key limits,
  permissions, or audit actor fields.
- Before closing a conjunctive task step, split "what this task ships" from "what
  the next task must consume"; if the latter is true, put it into the receiving
  task immediately.

## Action items

- [x] Handed stronger R2 confirmation to 20260729-125040 with an explicit DoD
      test: `test_one_way_action_requires_stronger_confirmation`.
