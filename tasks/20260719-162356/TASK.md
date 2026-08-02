# Integrate Codex agent backend via openai-codex Python SDK (subscription auth)

- PRIORITY: 20
- TAGS: feature, backlog, agent, llm
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Integrate OpenAI Codex as the Scufris agent backend via the official
`openai-codex` Python SDK, behind a small Scufris `Agent` interface, with
"Sign in with ChatGPT" (device-code) as the primary auth and an API key as a
fallback auth mode.

## Steps

- [x] Add `openai-codex` (pinned) via `uv add`; verify it imports in the rebuilt
      venv and introspect the real SDK surface (method names for auth /
      thread_start / run) rather than trusting the spike's paraphrase.
- [x] Define `scufris/agent.py`: an `AgentReply` pydantic model, an `Agent`
      Protocol (`async chat(prompt) -> AgentReply`), a `CodexAgent`
      implementation wrapping the SDK (lazy import so a missing SDK/binary never
      breaks app import; `sandbox="read-only"`, model from settings), a
      `DisabledAgent` stub (clear message when the agent is off/unconfigured),
      and a `build_agent(settings)` factory.
- [x] Auth: a `login(settings)` helper doing device-code (`login_chatgpt_*`)
      primary, `login_api_key` fallback; surface the device URL/code to the
      operator. Fail with a clear, actionable error when unauthenticated.
- [x] Config: extend settings with agent knobs (`agent_enabled`, `agent_model`,
      `agent_auth_mode` chatgpt|api_key, api key env for fallback, codex home);
      update `.env.example`.
- [x] CLI: add an argparse dispatch to `scufris` - default `serve`, plus
      `scufris login` (device-code) and `scufris chat "<prompt>"` (one turn) so
      the operator can verify the live path.
- [x] Tests: `build_agent` returns `DisabledAgent` when off; `DisabledAgent`
      reply; `CodexAgent.chat` with the SDK boundary MOCKED (patch `AsyncCodex`)
      asserting it runs a turn and returns the text; a clear error when the SDK
      is absent; config parsing. App still boots with the agent disabled.
- [x] Run `ruff check .`, `mypy .`, `pytest` green.

## Definition of Done

- An `Agent` protocol with a `CodexAgent` (openai-codex SDK) and a
  `DisabledAgent`, selected by a `build_agent` factory from settings; the SDK is
  lazily imported so the app runs with the agent disabled and no codex binary.
- `scufris login` and `scufris chat "..."` exist for the operator to exercise the
  real subscription path; the app boots unchanged when the agent is off.
- Tests green with the SDK boundary mocked/faked. ruff, mypy, pytest green.
- HONEST SCOPE: real `codex login` (device-code) and a live billed model call are
  the operator's to run (no ChatGPT credentials or codex binary in this env), so
  the live end-to-end call is documented, not automated here.

## Notes

- Spike: tasks/20260719-153040/SPIKE.md (recommends Codex + openai-codex SDK;
  records the subscription-auth ToS gray area - treat as personal, single-user,
  single-machine use).
- SDK: `pip install openai-codex` (via uv). `Codex()` / `AsyncCodex()` context
  manager, `thread_start(model=, sandbox=)`, `thread.run(prompt) -> TurnResult`;
  auth `login_chatgpt_device_code()` (headless primary), `login_api_key()`
  (fallback). Spawns a local `codex app-server` (JSON-RPC over stdio).
- Keep the harness behind a Scufris `Agent` protocol so the provider/auth is
  swappable (opencode or API-key path) without touching the chat UI or tools.
- Model default a config knob (target `gpt-5.5`; a GPT-5.6 tier if the plan
  exposes it). Pin the Codex + SDK versions (0.x, breaking changes).
- Confirm the streaming API for responsive chat during /plan.
- Depends on nothing hard; the chat panel (tatr 20260719-162406) and the MCP
  tool server (tatr 20260719-162419) build on this. ToS posture from the spike
  is a hard design constraint, not a footnote.

## Implementation

- `scufris/agent.py`: `Agent` protocol (`chat`/`aclose`), `AgentReply` model,
  `DisabledAgent` (raises `AgentUnavailable` with an actionable message),
  `CodexAgent` driving `openai_codex.AsyncCodex` (lazy import; one reused thread;
  `sandbox="read-only"`), a `build_agent(settings)` factory, and a `login()`
  helper (device-code primary via `login_chatgpt_device_code()`, API-key
  fallback). The SDK is reached through an injectable `open_client` seam so tests
  fake it - no SDK/binary/network needed.
- Coded against the SDK's REAL API, introspected from the installed wheel
  (`openai_codex.AsyncCodex.thread_start(...)` -> `thread.run(prompt)` ->
  `TurnResult.final_response`; `Sandbox.read_only`; device handle
  `.verification_url`/`.user_code`/`.wait()`), not the spike's paraphrase.
- `scufris/cli.py`: argparse dispatch - `serve` (default), `login`, `chat`;
  `scufris/__main__.py` now points at it; `app.main` renamed `run_server`.
- Config: agent knobs in `scufris/config.py` + `.env.example`; README gains an
  operator "The agent (optional)" section.
- Tests (`tests/test_agent.py`, `tests/test_cli.py`): factory selection, disabled
  raises, CodexAgent runs a turn + reuses the thread with a fake client, empty
  model -> None, the real opener raises `AgentUnavailable` when the SDK is absent,
  and CLI serve/chat dispatch. 15 tests, ruff+mypy+pytest green.

### Deviation and the NixOS blocker (important)

`openai-codex` is NOT a pinned dependency. It pulls a prebuilt `codex` CLI binary
wheel that fails to build in the uv2nix venv (auto-patchelf cannot satisfy
`libtinfo.so.6` for a bundled zsh), which would break `nix develop` and
`nix flake check` for the whole project. So the SDK is operator-installed and
lazy-imported; the app runs with the agent disabled by default and no codex
binary. Making the codex runtime actually work on NixOS (nix-ld / FHS / a nixpkgs
`codex`) is filed as a follow-up. HONEST SCOPE holds: real device-code login and
a live billed model call are unverified here (no credentials, no working binary
in this env) - only the mocked boundary is tested.
