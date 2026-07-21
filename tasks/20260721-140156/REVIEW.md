# Review - 20260721-140156 (Package web/dist as a Nix derivation)

## Round 1 (inline critical pass; reviewer had full context)

Diff: additive change to `flake.nix` only - a `scufrisWeb` derivation via
`pkgs.buildNpmPackage` and a `packages.web` output. No Python source touched.

### Checks

- `nix build .#web` -> `result/index.html`, all page bundles
  (agent/agents/projects/settings/stats), `agent-detail.html`, and the
  `projects/ agents/ settings/ stats/` partial dirs. Matches what `app.py`
  serves (StaticFiles mount with html=True; `/agents/<id>` -> agent-detail.html).
- Hermetic: builds from `web/src` (git-tracked), NOT the gitignored on-disk
  `web/dist`. The Nix output is a fresh production build, more current than the
  stale on-disk dist (it includes settings/stats bundles the old dist lacked).
- `nix flake check` passes (ruff/mypy/pytest gate unchanged - no Python touched).

### Findings

- [minor, accepted] `version = "0.1.0"` is hardcoded, duplicating
  `web/package.json`. Drifts if the frontend version bumps. Low impact (version
  is cosmetic for a static bundle); leaving it literal keeps the derivation
  self-contained. Note for a future cleanup, not a blocker.
- [minor, accepted] `.js.map` source maps ship in the output (~1.4MB). Harmless
  for a localhost single-operator dashboard; dropping them would need a webpack
  prod-mode devtool tweak, out of scope for plumbing.
- [info] `npmDepsHash` must be refreshed when `web/package-lock.json` changes
  (standard buildNpmPackage constraint; the fake-hash bootstrap is the refresh
  recipe).

- VERDICT: APPROVE

Additive, hermetic, verified by build + flake check. No correctness concerns.
