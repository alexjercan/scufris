# Retro: Make the Codex runtime work on NixOS

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- Exploration flipped the whole approach for the better: the operator already
  had nixpkgs `codex` on PATH + `~/.codex` auth, and `codex exec
  --output-last-message` returns a clean final message. Testing that FIRST (one
  tiny live call) turned a fuzzy "make the SDK runtime work" task into a simple,
  robust "shell out to codex exec" - no SDK, no bundled binary, no uv2nix surgery,
  no version skew.
- The end goal is actually PROVEN: `scufris chat` returned a real GPT-5.5 reply
  on this host. That is worth far more than green unit tests for an agent.
- The fake-`codex` script integration test exercises the real
  `create_subprocess_exec` path (output-file read, nonzero exit, missing binary)
  without the real binary - real coverage of the plumbing that matters.
- Swapping the implementation behind the unchanged `Agent` interface meant the
  chat panel and MCP tasks are unaffected; only agent.py's internals changed.

## What went wrong / friction

- The prior task built the whole SDK client path (openai-codex) that turned out
  un-installable on NixOS; this task discarded it. The spike had flagged the SDK
  as the recommendation, but did not check installability on the target host - a
  feasibility probe during the spike would have pointed at `codex exec` directly.
- Two dev-shell rebuilds cost a few minutes each (adding pkgs.codex, then the
  test-helper typing fix). Expected for nix.

## Lessons

- `probe-runtime-on-target-host-early`: for an external-tool integration, run the
  tool on the actual target host BEFORE designing around a specific client
  (SDK vs CLI). One `codex exec` live call reframed the whole task; the earlier
  spike's SDK recommendation was right on capability but wrong on NixOS
  installability.
- `codex-exec-is-the-nixos-path`: drive Codex via `codex exec --sandbox read-only
  --skip-git-repo-check --ephemeral --output-last-message <file>` (nixpkgs
  `codex`, shared `~/.codex` auth) - not the openai-codex SDK, whose bundled
  binary breaks the uv2nix venv.

## Follow-ups

- Multi-turn continuity for chat (drop `--ephemeral` + `codex exec resume`, or
  thread a session id) belongs to the chat-panel task (tatr 20260719-162406).
- Wiring `codex` into the packaged-app runtime closure (`nix run .#scufris`)
  rides with the web/dist packaging follow-up; today the app runs under
  `nix develop`.
