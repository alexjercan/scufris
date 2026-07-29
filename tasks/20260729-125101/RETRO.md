# Retro: publish a GitHub Release from a version tag

- DATE: 20260729
- TASK: 20260729-125101
- REVIEW ROUNDS: 2 (REQUEST_CHANGES, then APPROVE)

## What went well

- **Probing KVM instead of reasoning about it produced a third answer.** The
  DECISION had two branches: the runner has KVM (keep the VM test) or it does
  not (remove it). Reality was neither - `/dev/kvm` exists but is `root:kvm`
  0660 and unusable by the runner user. The obvious guard, `if [ -e /dev/kvm ]`,
  passes on this runner and then fails, and in skip-form would let a release
  publish having tested nothing. One 5-minute probe found that; no amount of
  reading would have.
- **The guard is one script, run identically by the operator and by CI.** It
  caught a genuinely dirty tree twice during development, which is the cheapest
  possible evidence that a check is load-bearing rather than decorative.
- **Reusing the previous task's plumbing.** Version agreement, notes extraction
  and pre-release classification all live in `release_tools.py` with tests, so
  the workflow contains no parsing of its own.

## What went wrong

- **The workflow shipped a blocker that local verification could not see.** All
  three jobs checked out `github.ref` - the branch a `workflow_dispatch` run
  starts from - so the "full gate on the TAGGED commit" and the smoke-tested
  wheel would both have come from master, and `gh release create` without
  `--verify-tag` would have invented the tag at the default branch head. The
  guard could not catch it because master's `pyproject.toml` matches the version
  being released. Everything I verified locally was true and none of it touched
  this.
- **Untrusted input reached the shell twice.** `${{ inputs.version }}` splices
  into the script body before bash ever sees it, so no amount of quoting inside
  the script helps; and `"$VERSION"` was spliced into a nested `bash -c '...'`.
  The reviewer reproduced execution in both positions.
- **The release was published before its assets were uploaded**, so an upload
  failure would leave a live, empty, watcher-notified release under a permanent
  version number. I had ticked the "leaves no half-created release" step while
  the code did the opposite.
- **I wrote a `| head` into a `set -o pipefail` script** - the repo's own
  "never let a pipe eat the exit code" rule, from the inside: SIGPIPE killed the
  script with a bare 141 before its own diagnostic printed.
- **The `docs/` scratch check contradicted AGENTS.md**, failing on any non-README
  file when AGENTS.md explicitly sanctions durable docs there. Fixing it
  properly meant naming the drawer (`docs/scratch/`) rather than guessing.

## Lessons

- `ci-jobs-must-pin-the-commit-not-the-ref`: a release workflow that resolves a
  tag NAME per job builds whatever that name points at when each job starts.
  Resolve once, emit the SHA, and have every later job check out the SHA - then
  assert the tag still names it. Otherwise a moved tag, or a branch sharing the
  tag's name, silently gates one commit and ships another.
- `never-interpolate-workflow-input-into-a-run-body`: `${{ }}` is expanded
  before bash sees the script, so it cannot be quoted. Pass untrusted values
  through `env:` and reference them as `"$VAR"`. Same inside a nested
  `bash -c '...'`: let the inner shell read the environment rather than
  splicing.
- `publish-last-create-as-a-draft`: any multi-step publish should create the
  artifact invisible (draft), fill it, and flip it visible in the final step.
  Otherwise a failure between "created" and "complete" leaves a live, empty
  thing that watchers were notified about and that a version number is now
  permanently attached to.
- `probe-produces-answers-a-decision-did-not-list`: this is
  `probe-runtime-on-target-host-early` sharpened. The probe's job is not to
  pick between the branches you wrote down - it is to find the branch you did
  not. Present-but-unusable was not on the list, and it is the case that breaks
  the obvious guard.

## What to do differently next time

Local verification proved the guard script, the build, and the smoke test -
everything that runs in a shell. It could not touch what only exists on a
runner: which ref is checked out, how `${{ }}` expands, what a partial failure
leaves behind. Next time, for CI work, enumerate the runner-only surfaces FIRST
and treat "I cannot test this locally" as the list of things to reason about
hardest and to hand to a reviewer explicitly - rather than reporting the parts
I could verify and letting the rest ride.
