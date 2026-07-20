# Review: Codex agent backend via openai-codex SDK

## Round 1 - 20260719

Scope: `scufris/agent.py`, `scufris/cli.py`, `scufris/app.py` (rename),
`scufris/__main__.py`, `scufris/config.py`, `.env.example`, `README.md`,
`tests/test_agent.py`, `tests/test_cli.py`, `pyproject.toml` (mypy override,
asyncio_mode).

### Correctness

- Coded against the SDK's REAL introspected API (`AsyncCodex.thread_start(...)`
  -> `thread.run(prompt) -> TurnResult.final_response`; device handle
  `.verification_url`/`.user_code`/`.wait()`), not the spike paraphrase - a good
  call that caught the real method names.
- The injectable `open_client` seam keeps the SDK/binary/network out of tests;
  15 tests cover factory selection, disabled-raises, a real turn + thread reuse,
  empty-model -> None, the SDK-absent error, and CLI dispatch. All green.
- Honest degradation verified by smoke: `scufris` still serves (200), `chat`
  with the agent off and `login` with the SDK absent both exit 1 with actionable
  messages.
- The right architectural call under the nix blocker: `openai-codex` is not a
  pinned dep (its bundled binary breaks the uv2nix venv), so the whole project's
  `nix develop` / `nix flake check` stay green while the agent stays optional.

### Observations

- MINOR / VERIFY-ON-FIRST-RUN: `thread_start` is called with `sandbox="read-only"`
  (a string), but the SDK types the arg as the `Sandbox` enum. If it does not
  coerce the string it will need `openai_codex.Sandbox.read_only` instead - a
  one-line change. Cannot be verified here (no runnable binary); flagged in the
  task's HONEST SCOPE. Same caveat for `final_response` being the assistant text.
- MINOR: `CodexAgent` starts/reuses one thread with no lock, so two concurrent
  `chat()` calls could race to start two threads. Fine for the single-user,
  sequential chat panel that consumes this next; a lock is a cheap follow-up if
  concurrency is added.
- MINOR: `_default_open_client` constructs `AsyncCodex()` (which spawns the
  app-server subprocess) without wrapping a launch failure into `AgentUnavailable`
  - a missing/unrunnable binary surfaces as a raw Codex/OS error. Acceptable, but
  wrapping would give the operator a cleaner message.

### Verdict

- VERDICT: APPROVE

The task meets its Definition of Done for the mock-verified scope: a
swappable `Agent` interface with a Codex implementation and disabled default, a
login/chat CLI for the operator, agent settings, and green checks with the SDK
faked. The live path (device-code login + a billed model call) is the operator's
to run, as scoped. The MINOR items are one-liners or belong to the follow-on chat
task; the sandbox/final_response assumptions are explicitly recorded for first
real run.
