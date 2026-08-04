# Make the Codex runtime work on NixOS (enable + live-verify the agent)

- PRIORITY: 18
- TAGS: feature, backlog, agent, nix
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

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

- [x] Add `pkgs.codex` to the flake dev shell so a known-good `codex` is on PATH
      under `nix develop` (note runtime/packaged-app PATH wiring as follow-up).
- [x] Rewrite `scufris/agent.py`: `CodexCliAgent` running `codex exec` via
      `asyncio` subprocess behind an injectable runner seam; resolve the binary
      from `settings.codex_bin or shutil.which("codex")`, raise `AgentUnavailable`
      when absent or on nonzero exit (surface stderr); read the reply from the
      `-o` file. Drop the SDK client/`login` internals.
- [x] Config: add `codex_bin` (optional override); keep `agent_enabled`,
      `agent_model`, `codex_home`; update `.env.example`.
- [x] CLI: `scufris login` now delegates to `codex login` (inherit stdio);
      `scufris chat` unchanged (uses the agent).
- [x] Tests: `CodexCliAgent.chat` with an injected fake runner; a real-subprocess
      test pointing `codex_bin` at a tiny fake `codex` script that writes the `-o`
      file (proves the plumbing); missing-binary -> `AgentUnavailable`; factory +
      disabled paths.
- [x] LIVE VERIFY on this host: `SCUFRIS_AGENT_ENABLED=1 scufris chat "..."`
      returns a real GPT-5.5 reply. Record the evidence.
- [x] Update README/AGENTS agent section (nixpkgs codex + `codex login`, drop the
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

## Implementation

- Exploration decided it: nixpkgs `codex` (on PATH, logged in) runs `codex exec
  --sandbox read-only --skip-git-repo-check --ephemeral --output-last-message
  <file> "<prompt>"` and returns a real reply. So the agent shells out to it -
  no openai-codex SDK, no bundled binary, no uv2nix surgery.
- `scufris/agent.py` rewritten: `CodexCliAgent` runs `codex exec` via an
  `asyncio` subprocess behind an injectable `runner` seam; `_run_codex_exec`
  resolves the binary (`settings.codex_bin or shutil.which("codex")`), enforces a
  timeout, raises `AgentUnavailable` on missing-binary / nonzero exit (surfacing
  stderr), and reads the reply from the `--output-last-message` file. `login()`
  now delegates to `codex login` (browser) / `codex login --with-api-key`. The
  SDK client path is gone; `Agent`/`AgentReply`/`DisabledAgent`/`build_agent`
  stay.
- `flake.nix`: `pkgs.codex` added to the dev shell. `config.py`: `codex_bin`,
  `agent_timeout_seconds`; `.env.example` + README agent section updated (nixpkgs
  codex + `codex login`, SDK framing removed). Dropped the now-unused
  `openai_codex` mypy override.
- Tests (`tests/test_agent.py`): runner-faked chat, factory + disabled paths, and
  REAL-subprocess integration against a tiny fake `codex` script (reads the `-o`
  file; nonzero exit -> AgentUnavailable; missing binary -> AgentUnavailable). 16
  tests, ruff+mypy+pytest green.

### Live verification (DoD)

On this NixOS host, in `nix develop` (codex 0.144.4 from `pkgs.codex`, existing
`~/.codex` auth): `SCUFRIS_AGENT_ENABLED=1 python -m scufris chat "Reply with
exactly one word: pong"` -> `pong`. Real GPT-5.5 reply, end to end. The
first-real-run assumptions from tasks/20260719-162356/REVIEW.md are resolved by
construction: `--sandbox read-only` is a valid CLI flag and the final message
comes from `--output-last-message`, not a guessed field.
