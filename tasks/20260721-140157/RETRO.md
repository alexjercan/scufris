# Retro - 20260721-140157 (Export homeManagerModules + nixosModules)

## What went well

- One shared module file parameterized by `isNixos` kept option declarations
  and the SCUFRIS_ env mapping in a single place; only the systemd surface
  (user unit schema vs `systemd.services` attrs) branches.
- Building a throwaway `home-manager.lib.homeManagerConfiguration` (pinned to
  the dotfiles' home-manager rev) and grepping the RENDERED unit file was a
  high-fidelity proof - far better than eval-only, and it caught the ExecStart
  list representation before it could surprise anyone.
- The flat `settings` attrset mapping to `SCUFRIS_<UPPER>` matched the
  pydantic-settings model exactly, so the whole config surface came for free.

## What went wrong / difficulties

- The nixos `DynamicUser` + default `Path.home()` state dir bug (see REVIEW):
  easy to miss because the HM path (real home) hides it. Caught by reasoning
  about the runtime, and will be pinned by the VM test.
- `home-manager` renders single-valued `Service.ExecStart` as a one-element
  list; `nix eval --raw` on it errors ("cannot coerce a list"). Use `--json` or
  `--apply builtins.head`, or just build the unit and read the file.

## Lessons

- `scufris-web-server-module-is-env-driven` (x1): the new scufris is ONE
  `scufris serve` web server configured entirely via `SCUFRIS_` env vars, not
  the old bot's server+bot split. A module maps a flat `settings` attrset to
  `SCUFRIS_<UPPER>` env, injects `SCUFRIS_WEB_DIST` from the `packages.web`
  derivation, and puts codex/claude/git on the service PATH (operator tools,
  not deps). 20260721-140157.
- `dynamicuser-needs-explicit-state-and-home` (x1): a systemd service with
  `DynamicUser=true` has no writable `$HOME`, so an app that defaults its state
  dir to `Path.home()/...` fails at runtime. Set `SCUFRIS_STATE_DIR`/`HOME` to
  the `StateDirectory` (`/var/lib/<name>`). The home-manager USER service is
  immune (real home) - the trap is nixos-system-service only. 20260721-140157.
- `render-hm-unit-file-not-eval` (x1): to verify a home-manager systemd unit,
  BUILD the `activationPackage` and read the generated `.service` file; eval of
  `Service.ExecStart` returns a one-element list that `--raw` refuses to coerce.
  20260721-140157.
