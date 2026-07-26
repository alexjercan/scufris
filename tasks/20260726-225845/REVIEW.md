# Review: today + SCUFRIS_DEN_PATH on the deployed scufris service

- TASK: 20260726-225845
- BRANCH: (nix.dotfiles) feature/scufris-today-den-path

## Round 1

- VERDICT: APPROVE
- REVIEWER: in-session (trivial diff: two lines of home-manager config, proven by
  building the real artifact and grepping the rendered unit; no scufris code change)

The change lives in nix.dotfiles (`home/alex/default.nix`, `programs.scufris`):
`pkgs.today` appended to `path`, and `den_path = "/home/alex/personal/the-den"`
added to `settings`. No scufris-repo code changed - the module already supports
both knobs generically.

Verification (load-bearing claim re-derived, not a finding): built
`homeConfigurations.alex.activationPackage` and read the generated
`scufris.service` (per lesson `render-hm-unit-file-not-eval`). The rendered unit
carries:
- `Environment=SCUFRIS_DEN_PATH=/home/alex/personal/the-den` (DoD 1);
- `PATH=...codex.../bin:...claude-code.../bin:...git.../bin:...today-0.1.0/bin:...`
  - `today` is on the service PATH alongside the existing binaries (DoD 1);
- the config evaluated and built clean (DoD 2).

Checks I ran:
- Confirmed the service does NOT otherwise get the den path: it does not inherit
  `home.sessionVariables.DEN_PATH` (systemd user services don't see HM session
  vars), so the explicit `settings.den_path` is genuinely required, not redundant.
- Confirmed scope: only the home-manager `programs.scufris` is configured (no NixOS
  `services.scufris`), so the HM edit is the whole surface.
- Confirmed the commit staged only `home/alex/default.nix` - the pre-existing dirty
  `flake.lock` was not swept in.

No BLOCKER/MAJOR/MINOR/NIT findings. No open `manual:` DoD items (both DoD proofs
are `cmd:`, run above). APPROVE.
