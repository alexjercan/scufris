# Make the Codex runtime work on NixOS (enable + live-verify the agent)

- STATUS: OPEN
- PRIORITY: 18
- TAGS: feature,backlog,agent,nix

## Goal

Make the Codex runtime actually work on this NixOS host so the Scufris agent can
be enabled and live-verified (a real GPT-5.5 reply), by driving the nixpkgs
`codex` CLI via `codex exec` instead of the SDK's bundled binary.

## Decision (from exploration, 20260719)

- nixpkgs has `codex` 0.142.2, already on the operator's PATH, and
  `~/.codex/auth.json` exists (logged in). Confirmed live: `codex exec -s
  read-only --skip-git-repo-check --ephemeral -o <file> "..."` returns a real
  model reply using that auth. This is the robust path - no `openai-codex` SDK,
  no bundled binary, no uv2nix surgery, no version skew.
- So: replace the SDK-based `CodexAgent` with a `CodexCliAgent` that shells out
  to `codex exec` (async subprocess, `-o` for the final message, `--json`
  available). Keep the same `Agent` interface, `DisabledAgent`, and factory. This
  supersedes the SDK client path from tatr 20260719-162356 (which cannot install
  on NixOS); the `codex-binary-breaks-uv2nix-venv` lesson stands as the reason.

## Steps

- [ ] Add `pkgs.codex` to the flake dev shell so a known-good `codex` is on PATH
      under `nix develop` (note runtime/packaged-app PATH wiring as follow-up).
- [ ] Rewrite `scufris/agent.py`: `CodexCliAgent` running `codex exec` via
      `asyncio` subprocess behind an injectable runner seam; resolve the binary
      from `settings.codex_bin or shutil.which("codex")`, raise `AgentUnavailable`
      when absent or on nonzero exit (surface stderr); read the reply from the
      `-o` file. Drop the SDK client/`login` internals.
- [ ] Config: add `codex_bin` (optional override); keep `agent_enabled`,
      `agent_model`, `codex_home`; update `.env.example`.
- [ ] CLI: `scufris login` now delegates to `codex login` (inherit stdio);
      `scufris chat` unchanged (uses the agent).
- [ ] Tests: `CodexCliAgent.chat` with an injected fake runner; a real-subprocess
      test pointing `codex_bin` at a tiny fake `codex` script that writes the `-o`
      file (proves the plumbing); missing-binary -> `AgentUnavailable`; factory +
      disabled paths.
- [ ] LIVE VERIFY on this host: `SCUFRIS_AGENT_ENABLED=1 scufris chat "..."`
      returns a real GPT-5.5 reply. Record the evidence.
- [ ] Update README/AGENTS agent section (nixpkgs codex + `codex login`, drop the
      bundled-SDK framing). `ruff`/`mypy`/`pytest` + `nix flake check` green.

## Definition of Done

- `SCUFRIS_AGENT_ENABLED=1 scufris chat "..."` returns a real model reply on this
  NixOS host (live-verified, evidence recorded); the app still runs with the
  agent off; `codex` is available under `nix develop`.
- Tests green with the subprocess boundary faked + a fake-codex integration test;
  ruff, mypy, pytest, and the intent of `nix flake check` green.

## Notes

- Supersedes the SDK client path in tatr 20260719-162356; the `Agent` interface,
  `AgentReply`, `DisabledAgent`, and `build_agent` factory are reused.
- Confirms/retires the first-real-run assumptions from
  tasks/20260719-162356/REVIEW.md (sandbox string, final-message source): with
  `codex exec` we pass `-s read-only` (a CLI flag, valid) and read the final
  message from `-o`, so both are resolved by construction.
- ToS posture stays personal, single-user (tasks/20260719-153040/SPIKE.md).
- Depends on the agent backend (tatr 20260719-162356, CLOSED).
