# Export homeManagerModules + nixosModules for the scufris web server

- STATUS: CLOSED
- PRIORITY: 11
- TAGS: infra, nix
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

The scufris flake exports only `packages`/`devShells`. The dotfiles consume
scufris via `inputs.scufris.homeManagerModules.default` (from the old
scufris-bot). To make the local repo the source of truth, THIS flake must
export `homeManagerModules.default` (primary, user-level per the user's
preference) and `nixosModules.default`, defining a `programs.scufris` for the
new single web-server (`scufris serve`, env-configured with the `SCUFRIS_`
prefix, default port 8000). The agent backends (codex/claude) are
operator-installed binaries on PATH, never Python deps (lessons
`codex-binary-breaks-uv2nix-venv`, `codex-exec-is-the-nixos-path`).

## Steps

- [x] Design the new `programs.scufris` interface for the web server:
      `enable`, a typed/opaque `settings` mapping to `SCUFRIS_*` env vars
      (host, port, stateDir, webDist, pollSeconds, agent knobs...),
      `environmentFile` for secrets (e.g. `SCUFRIS_OPENAI_API_KEY`), and an
      `extraPackages`/PATH list defaulting to include `pkgs.codex` (and claude
      if available) so the agent can shell out.
- [x] Add a `flake.homeManagerModules.default` that: installs the scufris
      package, defines a `systemd.user.services.scufris` running
      `${pkg}/bin/scufris serve`, sets `SCUFRIS_WEB_DIST` to the `web`
      derivation from task 1 (so the dashboard is served), maps settings to
      `Environment=`/`EnvironmentFile=`, and puts the agent binaries on the
      unit PATH.
- [x] Add a `flake.nixosModules.default` mirroring it as a system service
      (secondary; user prefers the home-manager one). Share the option/env
      mapping logic between the two where practical.
- [x] Because module options must resolve the per-system scufris + web
      packages, thread `self.packages.${pkgs.system}` into the modules (a
      `let`-bound module function or `self`-referencing module).

## Definition of Done

- `nix eval .#homeManagerModules.default --apply builtins.isAttrs` -> `true`
  and same for `.#nixosModules.default` (cmd).
- A throwaway home-manager eval that imports the module with
  `programs.scufris.enable = true` produces a `systemd.user.services.scufris`
  whose Environment includes `SCUFRIS_WEB_DIST=/nix/store/...web...` and whose
  ExecStart is `.../bin/scufris serve` (cmd: eval the config, grep the unit).
- `nix flake check` still passes.
- manual: the option surface reads sensibly for the user's existing config
  intent (host/port/agent), confirmed at Finish.
