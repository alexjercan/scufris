# Review - 20260721-140157 (Export homeManagerModules + nixosModules)

## Round 1 (inline critical pass; reviewer had full context)

Diff: new `nix/scufris-service.nix` (shared module, `isNixos` flag) + two
`flake.*Modules.default` outputs. Options: `enable`, `package`, `webPackage`,
`stateDir`, `settings` (flat attrset -> `SCUFRIS_<UPPER>`), `environmentFile`,
`path`.

### Verified

- Both outputs export: `nix eval .#homeManagerModules --apply builtins.attrNames`
  and `.#nixosModules` both -> `["default"]`.
- HM module rendered into a real `home-manager.lib.homeManagerConfiguration`
  and BUILT: the generated `~/.config/systemd/user/scufris.service` has
  `ExecStart=.../bin/scufris serve`, `SCUFRIS_WEB_DIST=/nix/store/...scufris-web...`
  (the T1 derivation), `SCUFRIS_HOST/PORT/AGENT_BACKEND`, a PATH with
  codex+claude-code+git+profile, `EnvironmentFile`, `Restart=on-failure`. This
  is the plumbing-only DoD proof - the packaged server WILL find the dashboard.
- `nix flake check` passes.

### Findings

- [major, FIXED] NixOS branch used `DynamicUser=true` while `state_dir` defaulted
  to `Path.home()/.local/state/scufris`; a DynamicUser has no writable home, so
  the store writes (agents.json/settings.json/projects.json) would fail at
  runtime. Fixed: the nixos branch defaults `SCUFRIS_STATE_DIR` and `HOME` to
  `/var/lib/scufris` (matches `StateDirectory=scufris`) unless the operator sets
  an explicit `stateDir`. The HM branch is unaffected (real user home).
- [minor, accepted] Agent under nixos `DynamicUser` still can't reach `~/.codex`
  / `~/.claude` (Path.home() -> /var/lib/scufris now, no auth there). Acceptable:
  the user prefers the HM user service (real home, real auth); the nixos system
  service is secondary and its agent backends are a documented operator concern.
- [minor, accepted] `settings` is a flat freeform attrset rather than fully
  typed options. Chosen deliberately: it mirrors pydantic-settings 1:1 (every
  `SCUFRIS_` field works with no module churn) and matches the app's own config
  model. Secrets are steered to `environmentFile` in the option description.
- [info] The nixos module's runtime behaviour is proven by the VM test filed as
  the follow-up task (user request) - this review covers the HM path live.

- VERDICT: APPROVE
