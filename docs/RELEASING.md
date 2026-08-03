# Releasing Scufris

Release sources must agree:

- `pyproject.toml`: version.
- `packages/*/pyproject.toml`: the SAME version. Workspace members ship as one
  artifact set, and the root wheel's `Requires-Dist: scufris-core` resolves only
  from the wheels attached beside it.
- `CHANGELOG.md`: dated, non-empty section.
- Git tag: `vX.Y.Z`.

Release only from the main checkout, on `master`, inside `nix develop`.

## Cut the release

```sh
cd ~/personal/scufris
nix develop
git branch --show-current
git pull --ff-only
```

Requires: `git branch --show-current` prints `master`.

1. Set `version = "X.Y.Z"` in `pyproject.toml` AND in every
   `packages/*/pyproject.toml`. `scripts/check-release-ready.sh` fails on a
   member left behind.
2. Cut and inspect the changelog.

```sh
scripts/cut-changelog.sh X.Y.Z
scripts/cut-changelog.sh --check X.Y.Z
scripts/release-notes.sh X.Y.Z
```

3. Commit the release metadata.

```sh
git commit -am "chore: release X.Y.Z"
scripts/check-release-ready.sh vX.Y.Z
```

The guard requires a clean tree. Fix failures and amend the release commit.

4. Push `master` before creating the tag.

```sh
git push origin master
git tag vX.Y.Z
git push origin vX.Y.Z
```

Push order is load-bearing. A rejected branch push must not leave a public tag
pointing outside `master`.

5. Watch the run for this tag.

```sh
gh run list --workflow release.yaml --branch vX.Y.Z
gh run watch --exit-status "$(gh run list --workflow release.yaml --branch vX.Y.Z --limit 1 --json databaseId --jq '.[0].databaseId')"
gh release view vX.Y.Z
```

Immediately after pushing, run lookup may return HTTP 404. Retry after GitHub
registers the run. `--branch` matches tag-triggered runs. For a manual run:
`gh run list --workflow release.yaml --event workflow_dispatch`.

## What the pipeline proves

- Release guard passes.
- `nix flake check` passes.
- `nix build .#scufris .#scufris-web` passes.
- NixOS VM and hostd VM tests pass with KVM.
- Every member's wheel and sdist build (`uv build --all-packages`).
- Clean install of the whole wheel set reports the tagged version.
- Draft release becomes public only after every earlier step passes.

## Version suffixes

- PEP 440 pre-release suffix: release page marked pre-release.
- Examples: `v0.2.0rc1`, `v1.0.0.dev4`.
- Post-release suffix: not a pre-release. Example: `v1.0.0.post1`.
- Changelog section: exact version, including suffix.

## Retry a runner failure

The publish job is idempotent and updates its existing draft.

```sh
gh run rerun "$(gh run list --workflow release.yaml --branch vX.Y.Z --limit 1 --json databaseId --jq '.[0].databaseId')" --failed
```

## Fix a repository failure before publication

Allowed only while the release was never public.

```sh
git push --delete origin vX.Y.Z
git tag -d vX.Y.Z
```

Delete an existing draft only when one exists:
`gh release delete vX.Y.Z --yes`.

Then fix, amend or commit, push `master`, create the tag again, and push it.

## Handle a bad published release

Alternatives, not a sequence:

- Preferred: fix forward with a new version. Mark the bad changelog section `[YANKED]`.
- Temporary demotion: `gh release edit vX.Y.Z --prerelease=true`. A workflow rerun resets this from the version classification.
- Remove page: `gh release delete vX.Y.Z --yes`. Tag remains resolvable.
- Remove tag: `git push --delete origin vX.Y.Z`. Only when nobody fetched it. Existing locked revisions survive; future tag resolution fails.

Never move a published tag.

## Consume a release

Pin the Nix flake input to a tag:

```nix
scufris.url = "github:alexjercan/scufris/vX.Y.Z";
```
