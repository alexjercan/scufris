# NixOS VM test for the scufris service (packages.vm-test)

- STATUS: CLOSED
- PRIORITY: 12
- TAGS: infra, nix, test
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

The user wants a NixOS VM test (like scufris-bot's old `nix/tests/scufris-vm.nix`)
so the deployment can be validated in a sandboxed VM instead of live on their
own config. It exercises the `nixosModules.default` end to end: boots a VM with
`services.scufris.enable = true`, waits for the unit, and asserts the dashboard
serves. This is the live-behavior proof the plumbing-only run otherwise skips -
but contained in a throwaway VM.

## Steps

- [x] Add `nix/tests/scufris-vm.nix` using `pkgs.testers.nixosTest`, importing
      `self.nixosModules.default` (the module already resolves the scufris + web
      packages from `self.packages.${system}`, so nothing extra is passed).
- [x] VM config: `services.scufris.enable = true`, `settings.host = "127.0.0.1"`,
      `settings.port = 8000`, `settings.agent_enabled = false` (no codex login
      in the VM), `curl` in systemPackages, small memory.
- [x] testScript: start; `wait_for_unit("scufris.service")`;
      `wait_for_open_port(8000)`; `curl /api/config` (liveness JSON);
      `curl /` and assert the dashboard `index.html` is served (proves
      SCUFRIS_WEB_DIST wiring - the whole point of the web derivation);
      restart-works check.
- [x] Expose it on Linux only as `packages.vm-test` (on-demand `nix build
      .#vm-test`), NOT in `checks` (keeps the light ruff/mypy/pytest gate fast).

## Definition of Done

- `nix build .#vm-test` passes (the VM boots, the unit is active, `/api/config`
  and `/` both respond, the served `/` contains the Scufris dashboard shell) -
  cmd.
- The nixos module's DynamicUser state-dir handling is exercised (the store
  writes succeed inside the VM).
- Not wired into `checks`, so `nix flake check` stays fast.
