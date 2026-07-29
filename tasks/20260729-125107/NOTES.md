# Notes: the first release

- DATE: 20260729
- TASK: 20260729-125107

## What v0.1.0 cost

Tag `v0.1.0` -> commit `339ea27`. Pipeline run 30449339746, green end to end in
about 7.5 minutes:

| Job | Duration |
|-----|----------|
| pre-release guard | 1m42s |
| full gate on the tagged commit (incl. the NixOS VM test) | 3m27s |
| build, smoke-test and publish | 2m09s |

Release page: https://github.com/alexjercan/scufris/releases/tag/v0.1.0 - two
assets (`scufris-0.1.0-py3-none-any.whl` 157897 bytes,
`scufris-0.1.0.tar.gz` 1431523 bytes), notes taken from the changelog's 0.1.0
section, not a draft, not marked pre-release.

Nothing broke. That is worth stating plainly: the release itself was boring,
which is what the three tasks before it were for. Everything that went wrong in
this epic went wrong during review, before anything was published.

## Verified independently of the green run

A green pipeline says the pipeline thinks it succeeded. These are checks made
from outside it:

- Downloaded the wheel FROM THE RELEASE PAGE into a scratch directory,
  installed it into a fresh `uv venv`, ran `scufris --version` -> `scufris
  0.1.0`. The artifact a stranger would download runs.
- `nix flake metadata github:alexjercan/scufris/v0.1.0` resolves to
  `339ea2794f417367abb3c4aa63611b1720370124`, and
  `nix eval github:alexjercan/scufris/v0.1.0#scufris.name` gives
  `scufris-0.1.0`. The tag a consumer pins is usable.
- Re-ran the whole workflow for the same tag through `workflow_dispatch` (run
  30449912896, green). Afterwards `gh release list` still showed exactly ONE
  release, with the same two assets and the same `publishedAt` timestamp
  (12:01:05Z) - it updated in place rather than duplicating. That is the
  idempotence DoD item, and it also exercised the dispatch code path that
  carried the round-1 blocker.

## The dotfiles pin

`~/personal/nix.dotfiles` commit `0b20eb8` (LOCAL ONLY - not pushed, and the
system was NOT rebuilt, per the operator's instruction):

- `flake.nix`: `scufris.url` moves from `github:alexjercan/scufris` to
  `github:alexjercan/scufris/v0.1.0`.
- `flake.lock`: records `ref: v0.1.0` at rev `339ea27...`.

OUTSTANDING for the operator: rebuild against the pinned input and confirm the
running dashboard reports `0.1.0`. That is the epic's manual acceptance item and
it cannot be closed from here - what is proven so far is that the packaged
artifact reports 0.1.0, not that the running service does.

## Leftovers cleaned up

`sprout land` removes the local branch and worktree, but two branches had been
PUSHED to origin during this epic to gather CI evidence (`infra/ci-push-pr` for
PR #1, `infra/release-from-tag` for the KVM probe). Landing squash-merges, so
neither showed as merged, and both lingered as stale remote snapshots - the
release one still carrying the temporary `kvm-probe.yaml` with an `on: push`
trigger. Both deleted after confirming every change on them was either on
master or deliberately dropped. PR #1 was closed unmerged.

Lesson for next time: a branch pushed for evidence needs deleting explicitly;
the landing tool will not do it, because from its point of view the branch is
already gone.

## What the next release should be able to skip

Everything in `AGENTS.md`'s Releasing section was executed literally for this
one and worked. The parts that needed thought this time and should not next
time: whether the VM test can run on a hosted runner (it can, 102s, after a
chmod), what a release artifact is (wheel + sdist, no PyPI), and whether the
changelog cut belongs to the tagging task (no - it belongs wherever makes the
agreement test assert something real).
