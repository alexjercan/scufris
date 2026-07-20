# Review: Make the Codex runtime work on NixOS

## Round 1 - 20260719

Scope: `scufris/agent.py` (rewrite), `scufris/config.py`, `flake.nix`
(`pkgs.codex`), `.env.example`, `README.md`, `tests/test_agent.py`,
`pyproject.toml` (drop openai_codex mypy override).

### Correctness

- The core goal is met and PROVEN live: `SCUFRIS_AGENT_ENABLED=1 scufris chat` ->
  a real GPT-5.5 reply on this NixOS host, via nixpkgs `codex exec` and the
  existing `~/.codex` auth. This is the decisive evidence a build-only check
  would have missed.
- The approach is a clear improvement over the SDK path: no un-installable
  `openai-codex`, no bundled binary, no uv2nix surgery, no 0.142-vs-0.144 skew -
  it drives the exact binary the operator validated.
- Subprocess handling is sound: `create_subprocess_exec` (no shell), read-only
  sandbox, `stdin=DEVNULL`, a timeout with kill-on-expiry, nonzero exit ->
  `AgentUnavailable` with stderr, reply read from `--output-last-message`.
- Real subprocess coverage: the fake-`codex` integration tests exercise the
  actual `create_subprocess_exec` plumbing (output-file read, nonzero exit,
  missing binary) - not just a mocked runner. 16 tests, all green; the app still
  runs with the agent off.
- The first-real-run assumptions from the prior task are retired by construction
  (a CLI flag and an output file, not guessed SDK fields).

### Observations (non-blocking)

- MINOR: `--ephemeral` means each `chat` is a fresh turn with no memory. Correct
  for this "make it work + verify" milestone; multi-turn continuity (drop
  `--ephemeral` + `codex exec resume`, or thread a session id) belongs to the
  chat-panel task (20260719-162406). Worth an explicit note there.
- MINOR: `pkgs.codex` is only in the dev shell, so `nix run .#scufris` (packaged
  app) would not find `codex` on PATH. Fine now (the app runs under `nix
  develop`); wiring codex into the runtime closure rides with the existing
  web/dist packaging follow-up.
- NIT: dev-shell codex is 0.144.4 (from nixpkgs pin) while the operator's
  profile codex is 0.142.2; both share `~/.codex` auth and both work. `codex_bin`
  lets the operator pin a specific one.

### Verdict

- VERDICT: APPROVE

The task meets its Definition of Done - the agent is live-verified
returning a real model reply on this host, the app is unharmed with the agent
off, `codex` is in the dev shell, and checks are green with real subprocess
coverage. The MINOR items are scoped to later tasks.
