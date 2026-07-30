# Put nvidia-smi on the deployed scufris service PATH so GPU stats appear

- STATUS: CLOSED
- PRIORITY: 30
- TAGS: bug,nix,deploy,gpu
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Flow State

- WORKING NOTE: nix.dotfiles-only fix (scufris repo gets only this task record),
  mirroring the today (20260726-225845) and macros (20260727-011526) deploys.

## Goal

In the deployed scufris (home-manager user service on the NixOS host) the `stats`
page shows `gpus: []`, while the local dev run on :7000 shows the GPU. Make the
GPU appear in the deployed dashboard.

## Understanding (verified 2026-07-27)

Root cause: `nvidia-smi` is not on the deployed service PATH.

- GPU stats come from `nvidia-smi` via `scufris/metrics.py:149`
  (`shutil.which("nvidia-smi")`); when it is not found the runner returns None
  and `parse_gpus(None)` yields `[]`.
- `nvidia-smi` lives at `/run/current-system/sw/bin/nvidia-smi`, a symlink into
  the NixOS system profile driver `nvidia-x11-595.84-bin` (store path
  `/nix/store/2fhwk74jlglx2h1vn8xjfygyw18bs1cb-nvidia-x11-595.84-bin`).
- scufris is deployed as a home-manager USER service (`programs.scufris`,
  nix.dotfiles home/alex/default.nix). The service module (scufris
  nix/scufris-service.nix, HM branch) OVERRIDES PATH entirely:
  `Environment = ... "PATH=${makeBinPath cfg.path}:${profileDir}/bin"`. The
  current rendered PATH is codex:claude:git:today:macros:~/.nix-profile/bin and
  does NOT include `/run/current-system/sw/bin`, so `nvidia-smi` is unreachable.
- Locally on :7000 the app runs from an interactive shell whose PATH includes
  `/run/current-system/sw/bin`, so `nvidia-smi` resolves and the GPU shows.
- This is the same class as the today/macros "on the deployed service PATH"
  deploys, and the user chose the same fix shape: add the driver bin package to
  `programs.scufris.path` in nix.dotfiles.

Version-match proof (the crux): nvidia-smi/NVML must match the LOADED kernel
module version or it errors "Driver/library version mismatch". Both the NixOS
host and the HM `pkgs` resolve from the SAME `nixpkgs` input, so
`pkgs.linuxPackages.nvidia_x11.bin` evaluates to the identical store path as the
running driver:
`nix eval` -> `/nix/store/2fhwk74jlglx2h1vn8xjfygyw18bs1cb-nvidia-x11-595.84-bin`
(matches the running `/run/current-system/sw/bin/nvidia-smi` target). So the
`nvidia-smi` we add is exactly the one that matches the loaded module.

## Steps

- [x] In nix.dotfiles `home/alex/default.nix`, append `pkgs.linuxPackages.nvidia_x11.bin`
      to `programs.scufris.path` -> `[pkgs.codex pkgs.claude-code pkgs.git
      pkgs.today pkgs.macros pkgs.linuxPackages.nvidia_x11.bin]`, and extend the
      adjacent comment to note nvidia-smi backs the GPU `stats` page.
      Done on nix.dotfiles branch `feature/scufris-gpu-service-path`.
- [x] Prove it: build `homeConfigurations.alex.activationPackage` and grep the
      rendered `scufris.service` `PATH=` for the nvidia-x11 store path (and
      confirm codex/claude/git/today/macros are all still present).

## Verification (2026-07-27)

- DoD1: rendered `scufris.service` `Environment=PATH=` now contains
  `/nix/store/2fhwk74jlglx2h1vn8xjfygyw18bs1cb-nvidia-x11-595.84-bin/bin`
  alongside codex/claude/git/today/macros (all still present).
- DoD2: `nix build .#homeConfigurations.alex.activationPackage` -> exit 0.
- DoD3: added store path == `readlink -f /run/current-system/sw/bin/nvidia-smi`
  target (`...2fhwk74jl...-nvidia-x11-595.84-bin/bin/nvidia-smi`) -> no NVML
  version mismatch.
- End-to-end: running that exact `nvidia-smi` with the app's `_GPU_QUERY`
  returns valid 9-field CSV (`NVIDIA GeForce RTX 3060 Ti, 2, 305, 8192, 39,
  9.94, 225.00, 210, 405`), which `parse_gpus` populates into one GpuStats.
- Deploy: takes effect after `home-manager switch` (operator's call).

## Definition of Done

1. The rendered scufris HM unit has the nvidia-x11 bin on its `PATH=` (alongside
   the existing codex/claude/git/today/macros).
   (cmd: build the HM config, grep the generated scufris.service PATH= for `nvidia-x11`)
2. The HM config still builds clean.
   (cmd: `nix build .#homeConfigurations.alex.activationPackage`)
3. The store path added equals the running driver's, so nvidia-smi will not hit
   an NVML version mismatch. (cmd: compare the grepped store path to
   `readlink -f /run/current-system/sw/bin/nvidia-smi`)

## Notes

- The scufris repo change is ONLY this task record (like today/macros). The code
  change lands in nix.dotfiles.
- nvidia-smi self-resolves against the loaded kernel module; no SCUFRIS_* knob
  needed (unlike den_path).
- Landing: merge+switch (`home-manager switch`) is the operator's call; flow does
  not push or deploy.
