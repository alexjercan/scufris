# Make the Codex runtime work on NixOS (enable + live-verify the agent)

- STATUS: OPEN
- PRIORITY: 18
- TAGS: feature,backlog,agent,nix

## Goal

Make the OpenAI `codex` runtime actually run on this NixOS host so the Scufris
agent can be enabled and live-verified (real device-code login + a billed model
call).

## Notes

- From tasks/20260719-162356 (agent backend) RETRO/REVIEW: `openai-codex` bundles
  a prebuilt `codex` CLI binary that fails auto-patchelf in the uv2nix build
  (`libtinfo.so.6` for a bundled zsh), and the dev venv uses `deps.all`, so the
  SDK cannot be a pinned/optional dep. The agent code is ready and
  operator-installed; this task makes the binary runnable.
- Options to explore: enable `programs.nix-ld` (or an FHS `buildFHSEnv`) so the
  downloaded ELF runs; OR package `codex` from nixpkgs (if available) / build the
  Rust CLI in the flake and point the SDK at it; OR run the agent in a container.
- Once runnable: verify `uv pip install openai-codex` + `scufris login` +
  `scufris chat "..."` end to end against the operator's ChatGPT subscription.
  Confirm the first-real-run assumptions from tasks/20260719-162356/REVIEW.md:
  `sandbox="read-only"` string vs the `Sandbox` enum, and `final_response` being
  the reply text - each a one-line fix in scufris/agent.py if wrong.
- ToS posture stays personal, single-user (tasks/20260719-153040/SPIKE.md).
- Depends on the agent backend (tatr 20260719-162356, CLOSED).
