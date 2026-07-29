# Retro: EPIC - Ship tagged releases from CI

- DATE: 20260729
- TASK: 20260729-124706
- CHILD TASKS: 4, all CLOSED and landed
- REVIEW ROUNDS ACROSS THE EPIC: 7 (5 REQUEST_CHANGES, 2 straight APPROVE)

## What the epic actually delivered

A project with ~480 commits, no CI, no tags and a changelog that had never been
closed now has: a gate that runs on every push and PR and is proven to go red;
one source of version truth with the plumbing to keep it that way; a tag-driven
release pipeline that verifies the tagged commit and proves the artifact runs
before publishing; and v0.1.0 on the releases page with a consumer pinned to it.

## The pattern across all four tasks

Every single REQUEST_CHANGES finding in this epic - without exception - was the
same shape: **code that reported success while doing the wrong thing.**

- `nix flake check` passing while `nix build .#web` was broken for consumers,
  because `checks` are built and `packages` only evaluated.
- A changelog cut that wrote a file with no released section and no links, and
  printed "cut CHANGELOG.md for 1.0.0", exit 0.
- Three tests named in a Definition of Done that compared the app's version
  against the same call the app makes, so they were green when everything
  reported `0.0.0+unknown`.
- A test written to pin the code-fence fix whose fixture never reached the
  fence code - it passed with fence detection switched off entirely.
- A release workflow that would have checked out master, passed the guard, and
  invented a tag at the default branch head.
- A `docs/` check that would have failed on the first legitimate design doc.
- A documented procedure that would have tagged a feature branch.

None of these were caught by a failing check, because by construction none of
them fail. They were caught by out-of-context review - and two of them by the
reviewer REVERTING a fix and confirming the new test failed. That technique
found a regression I had introduced with a fix, and a test that proved nothing.

## What went well

- **Probing the runner instead of reasoning about it**, early and repeatedly.
  The first cold CI run answered the epic's biggest open question (Nix on a
  hosted runner) in under two minutes and made the "no binary cache" decision
  evidence-based. The KVM probe found an answer neither branch of the DECISION
  had imagined: `/dev/kvm` present but unusable, which makes the obvious
  `if [ -e /dev/kvm ]` guard pass and then fail.
- **Proving each gate RED before trusting it green.** A deliberate break for
  CI; a corrupted task record for the conformance check; a crafted version
  mismatch for the release guard; a dispatch with a nonexistent tag for the
  pipeline. A gate only ever observed passing has not been observed.
- **Cutting the changelog in the version task rather than the tagging task**,
  against the plan. It made the epic's named agreement test assert a real fact
  three tasks earlier, and forced idempotence to be a real property.
- **Verifying from outside the thing that claims success**: downloading the
  published wheel and running it, resolving the flake pin with `nix flake
  metadata`.

## What went wrong

- **I reported "verified locally" as though it covered the work.** For the
  release pipeline it covered the guard, the build and the smoke test, and
  touched none of what only exists on a runner: which ref is checked out, how
  `${{ }}` expands, what a partial failure leaves behind. That is where the
  blocker was.
- **Two of my own fixes created new defects** - automatic re-dating broke the
  idempotence it was meant to serve, and a fence test that never ran the fence
  code. Both found by review, not by me.
- **I left two pushed branches behind.** `sprout land` cleans up locally, so
  branches pushed for CI evidence survived, never showed as merged, and one
  still carried a temporary probe workflow. The operator found them.
- **Planning drifted from reality in one place worth naming**: the plan put the
  changelog cut in the last task. Moving it was right, but it meant the last
  task's step list described work already done, which I had to annotate rather
  than silently tick.

## Lessons already in the ledger

Eleven entries were added across the four tasks. The ones that generalise
beyond this epic: `revert-the-fix-to-prove-the-test`,
`dod-named-tests-deserve-the-most-scrutiny`,
`a-fix-can-break-the-property-it-was-protecting`,
`prove-a-new-gate-red-before-trusting-it-green`,
`probe-produces-answers-a-decision-did-not-list`,
`write-a-procedure-in-failure-order-not-thought-order`.

## What to do differently next epic

For infrastructure work, write down FIRST the list of surfaces that cannot be
verified locally, and treat that list as the review brief. Every blocker in
this epic lived on that list: runner ref resolution, template expansion,
partial-failure states, KVM availability. The parts I could run locally were
fine every time.

And when a review asks for an escape hatch, add the hatch - do not change the
default. That single mistake cost a whole review round.
