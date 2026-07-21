# Review - 20260721-141458 (NixOS VM test for the scufris service)

## Round 1 (inline critical pass; reviewer had full context)

Diff: new `nix/tests/scufris-vm.nix` (`pkgs.testers.nixosTest` importing
`nixosModules.default`) + a Linux-only `packages.vm-test` output.

### Verified - `nix build .#vm-test` passed (exit 0), all 5 subtests green:

- `scufris.service` reached active; TCP 8000 open.
- `GET /` -> 200 serving the built dashboard (`Scufris` + `id="app"` asserted) -
  the SCUFRIS_WEB_DIST -> packages.web wiring proven live, not just in eval.
- `GET /api/config` -> 200 with `"agent_enabled":false`.
- `test -d /var/lib/scufris` succeeds - the DynamicUser StateDirectory + the
  SCUFRIS_STATE_DIR/HOME override from T2's fix are exercised; state writes work.
- `systemctl restart` brings the unit back and it still serves.

### Findings

- [minor, FIXED during work] The new test file was untracked, so the dirty-tree
  flake couldn't see it (`Path ... is not tracked by Git`). `git add` of the
  explicit path (never `-A` here - the node_modules-symlink lesson) fixed it.
- [decision] Exposed as `packages.vm-test`, NOT a `checks` entry, so the light
  ruff/mypy/pytest gate stays fast; the VM (KVM + full image boot) runs on
  demand via `nix build .#vm-test`. Guarded with `lib.optionalAttrs
  pkgs.stdenv.isLinux` so darwin systems still evaluate.
- [minor, accepted] Agent path not exercised (agent_enabled = false): a VM has
  no codex/claude login. Chat has ample unit coverage; matches the old
  scufris-bot VM test's scope decision.

- VERDICT: APPROVE
