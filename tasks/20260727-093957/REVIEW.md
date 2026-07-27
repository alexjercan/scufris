# Review: nvidia-smi on the deployed scufris service PATH

- VERDICT: APPROVE
- TASK: 20260727-093957
- SCOPE: nix.dotfiles branch `feature/scufris-gpu-service-path` (one-line
  `programs.scufris.path` addition + comment) and scufris task record.

## Round 1 (out-of-context reviewer) - APPROVE

Verdict: APPROVE. All load-bearing claims verified against the live system.

Verified:
- `pkgs.linuxPackages.nvidia_x11.bin` == the running driver store path
  (`...2fhwk74jl...-nvidia-x11-595.84-bin`); provides nvidia-smi. No NVML
  mismatch in the current config.
- Independently cross-checked: the host's resolved driver
  `config.hardware.nvidia.package.bin` (from `boot.kernelPackages.nvidiaPackages.stable`)
  ALSO resolves to the same store path - the two attribute paths genuinely
  converge, not just "same nixpkgs input" hand-waving.
- `boot.kernelPackages` is not overridden in hosts/ or home/; `open = false`,
  `nvidiaPackages.stable` in effect. Divergence scenarios do not currently apply.
- No shadowing: nvidia_x11.bin/bin contains only `nvidia-*` binaries, appended
  LAST after codex/claude/git/today/macros. Safe ordering.

Findings addressed:
- MEDIUM (version-match invariant is real but was framed as "same nixpkgs input"
  which understates that it also depends on the host running the default kernel +
  stable driver): FIXED - rewrote the `home/alex/default.nix` comment to state the
  default-kernel/stable-driver assumption and what to change if the host pins a
  non-default kernel or beta/legacy/production nvidia package.
- LOW (no shadowing): acknowledged, no change needed.
- NIT (comment overstated guarantee): folded into the MEDIUM fix.
- NIT (TASK.md STATUS still IN_PROGRESS mid-flow): expected; closed at land.

No CRITICAL or HIGH issues. The comment-only fix does not change the rendered
PATH, so DoD proofs still hold.
