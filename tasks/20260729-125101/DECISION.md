# Decision: a release is a tag plus wheel and sdist; no PyPI, and the VM test guards the release only

- DATE: 20260729-130000
- STATUS: ACCEPTED
- TASK: 20260729-125101
- TAGS: decision, release, ci, packaging, v0.1.0

## Context

The epic left "what a Scufris release artifact IS" open. There are two distinct
consumers and they want different things:

- `~/personal/nix.dotfiles` consumes the project as a flake input and needs
  only a tag to pin (`github:alexjercan/scufris/v0.1.0`). It downloads nothing
  from the release page.
- A person told "go look at this project" wants a release page with something
  on it and notes that say what changed.

Publishing to PyPI is a third, materially different thing: it is irreversible
per version, needs a PyPI project and trusted-publisher configuration to exist
before the first tag, and turns every tag into a public package release.

Separately, `nix build .#vm-test` is deliberately outside `checks` in
`flake.nix` because it needs KVM, and hosted-runner KVM availability is not
something this plan can assert from a local shell.

## Decision

A release is:

1. The git tag itself, which is what the flake consumer pins.
2. A wheel and an sdist built with `uv build`, attached to the GitHub Release.
3. Release notes that are exactly that version's `CHANGELOG.md` section.

The wheel is smoke-tested BEFORE publishing: install it into a throwaway
virtualenv and run `scufris --version`, asserting it prints the tagged version.
A release that cannot run does not get published.

Nothing is published to PyPI. If that is ever wanted it is its own task, with
its own decision, because it adds an irreversible outward-facing step.

The NixOS VM test is attempted in the RELEASE pipeline only, never in per-push
CI. If the hosted runner has no `/dev/kvm`, the step is REMOVED and the finding
is written into this task record. It is not left as a step that skips itself
and reports success - a green check that checked nothing is worse than an
absent one.

**RESOLVED by measurement (probe run 30446677138).** `ubuntu-latest` DOES have
`/dev/kvm`, but as `root:kvm` mode 0660 with the runner user outside the `kvm`
group - present and unusable at the same time. After `sudo chmod 666 /dev/kvm`
the test runs and passes in 102 seconds. So the step STAYS, unconditionally.

This is worth recording because the obvious guard is wrong here: an
`if [ -e /dev/kvm ]` check passes on this runner and then fails to use the
device, and its skip-form would let a release publish having tested nothing.
The workflow therefore fixes the permission and runs the test outright, so
losing KVM turns the release red rather than quietly hollowing it out.

## Alternatives considered

- **Also publish to PyPI** - rejected for v0.1.0: irreversible, needs operator
  setup before the first tag, and neither consumer above asked for it.
- **Flake only, no attached files** - enough for nix.dotfiles, but leaves the
  release page with nothing downloadable and no proof the distribution builds.
  Rejected: the epic's manual acceptance is "a page the operator would point
  another person at".
- **VM test in per-push CI** - rejected: it is the slowest check in the repo
  and its value is per-release, not per-commit.

## Consequences

Easier: releasing stays reversible (a bad release can be deleted and the tag
moved, since nothing was pushed to an immutable index), and the release proves
its own artifact runs. Harder: consumers who want `pip install scufris` cannot;
they install from the attached wheel or the flake. The VM test's fate is
decided by a real runner rather than by this document, so this decision is
knowingly incomplete on that point and the task must close the gap.
