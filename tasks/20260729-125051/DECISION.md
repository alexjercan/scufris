# Decision: CI runs the real flake via the Determinate installer, no third-party binary cache

- DATE: 20260729-130000
- STATUS: ACCEPTED
- TASK: 20260729-125051
- TAGS: decision, ci, nix, v0.1.0

## Context

`nix flake check` is the documented source of truth for green in this repo
(AGENTS.md, "Build, run, test"): it runs ruff, mypy and pytest each against a
fresh writable copy of the tree through `mkCheck` in `flake.nix`. Running that
on a GitHub-hosted runner needs Nix installed with flakes enabled, and the
uv2nix-built Python set is not in any public binary cache, so the first run
pays a cold build of whatever nixpkgs does not already provide.

Three shapes were on the table, and they are mutually exclusive in what they
demand from the operator before CI can ever be green:

- Determinate installer with no extra cache: needs nothing from the operator.
- Cachix: needs a Cachix cache created and a `CACHIX_AUTH_TOKEN` repo secret to
  exist BEFORE the first run, otherwise the push step fails on every build.
- Skipping Nix entirely (install Python/uv directly): fastest, but then CI
  never evaluates `flake.nix`, and the local gate and the CI gate can drift
  without anything noticing - which is the exact failure mode this epic exists
  to end.

## Decision

Use `DeterminateSystems/nix-installer-action` and run `nix flake check`
directly, relying on `cache.nixos.org` alone. No Cachix, no FlakeHub, no
secrets, no operator setup.

The cold and warm wall-clock costs are MEASURED and written into this task
record rather than assumed. If a warm run is slow enough to discourage small
pushes (the task's fourth DoD item), that becomes a new, separately prioritized
task to add a binary cache - not a silent widening of this one.

The frontend gate (`npm ci` && `npm run ci`) runs as its own job with
`actions/setup-node` caching, not through Nix: `web/` is an ordinary npm
project and the flake's `web` package only builds the static assets.

## Alternatives considered

- **Cachix from the start** - fastest warm builds, but it front-loads manual
  account setup onto a task whose whole point is that the gate stops depending
  on a human remembering something. Rejected for now; re-openable with real
  timing numbers from this task as the argument.
- **Drop Nix, run ruff/mypy/pytest on a plain Python setup** - rejected: it
  makes CI a different gate from the documented one, so `nix flake check` could
  break on master and CI would stay green.

## Consequences

Easier: CI is green on day one with nothing configured outside the repository,
and it exercises the actual flake consumers use. Harder: cold runs pay a full
uv2nix build, and a nixpkgs bump can make one run unusually long. That cost is
recorded, not hidden, so the follow-up decision has evidence behind it.

## Related

The NixOS VM test (`nix build .#vm-test`) stays OUT of this workflow - see
`tasks/20260729-125101/DECISION.md`.
