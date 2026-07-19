# Retro: Codex agent backend via openai-codex SDK

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- Introspecting the actually-installed SDK wheel (no-deps, into a throwaway dir)
  gave the REAL API surface, so `CodexAgent` was coded against
  `thread_start`/`thread.run`/`TurnResult.final_response` and the device-code
  handle's true attributes - not the spike's paraphrase, which differed in
  details (e.g. `run(input)` not `run(prompt=)`, `Sandbox` enum values).
- The injectable `open_client` seam made the whole thing testable with a fake
  client - 15 tests, zero SDK/binary/network - which is exactly the harness-first
  posture given the live path can't run here.
- Keeping `openai-codex` out of the pinned deps kept `nix develop` /
  `nix flake check` green for the whole project while still shipping a working,
  operator-enablable agent. The disabled-by-default smoke (serve still 200, chat
  and login degrade with clear messages) proved the app is unharmed.

## What went wrong / friction

- `uv add openai-codex` broke the dev shell: the SDK pulls a prebuilt `codex`
  binary wheel that fails auto-patchelf in uv2nix (`libtinfo.so.6` for a bundled
  zsh). Worse, because the dev venv uses `deps.all`, even an OPTIONAL extra would
  re-break it - so the dep can't live in pyproject at all on this nix setup.
  Recovered by `git checkout -- pyproject.toml uv.lock` (the add was uncommitted)
  since `uv remove` couldn't run once the shell wouldn't build.
- That is a real blocker for the recommended approach on NixOS: the codex binary
  needs nix-ld / an FHS env / a nixpkgs `codex` to run at all. Filed as a
  follow-up; the live agent is unverifiable until it's resolved.
- The `sandbox="read-only"` string vs the `Sandbox` enum, and `final_response`
  being the reply text, are assumptions I could not verify without a runnable
  binary - recorded as first-real-run checks, each a one-liner to fix.

## Lessons

- `codex-binary-breaks-uv2nix-venv`: `openai-codex` bundles a prebuilt `codex`
  CLI that fails auto-patchelf in the uv2nix build; keep it operator-installed and
  lazy-imported, never a pinned (or even optional) dep, because the dev venv uses
  `deps.all`. A NixOS runtime (nix-ld/FHS/nixpkgs codex) is a separate concern.
- `introspect-sdk-not-spike-paraphrase`: for a post-cutoff SDK, install the wheel
  no-deps and `dir()`/`inspect.signature` the real classes before coding against
  it - the spike's method names were close but wrong in specifics.
- `optional-dep-vs-deps-all`: because uv2nix's dev venv is built from
  `workspace.deps.all`, a dependency that must NOT be in the venv cannot be a
  pyproject optional-extra either; it has to stay out of the workspace entirely.

## Follow-ups

- Filed: make the Codex runtime work on NixOS (nix-ld / FHS / package `codex` in
  the flake) so the agent can actually run and be live-verified - see the new
  backlog task. Also carry the sandbox-enum and final_response first-run checks
  into whoever first runs it live (recorded in REVIEW.md / the task).
