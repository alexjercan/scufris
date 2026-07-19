# Integrate Codex agent backend via openai-codex Python SDK (subscription auth)

- STATUS: OPEN
- PRIORITY: 20
- TAGS: feature,backlog,agent,llm

## Goal

Integrate OpenAI Codex as the Scufris agent backend via the official
`openai-codex` Python SDK, behind a small Scufris `Agent` interface, with
"Sign in with ChatGPT" (device-code) as the primary auth and an API key as a
fallback auth mode.

## Steps

- [ ] Add `openai-codex` (pinned) via `uv add`; verify it imports in the rebuilt
      venv and introspect the real SDK surface (method names for auth /
      thread_start / run) rather than trusting the spike's paraphrase.
- [ ] Define `scufris/agent.py`: an `AgentReply` pydantic model, an `Agent`
      Protocol (`async chat(prompt) -> AgentReply`), a `CodexAgent`
      implementation wrapping the SDK (lazy import so a missing SDK/binary never
      breaks app import; `sandbox="read-only"`, model from settings), a
      `DisabledAgent` stub (clear message when the agent is off/unconfigured),
      and a `build_agent(settings)` factory.
- [ ] Auth: a `login(settings)` helper doing device-code (`login_chatgpt_*`)
      primary, `login_api_key` fallback; surface the device URL/code to the
      operator. Fail with a clear, actionable error when unauthenticated.
- [ ] Config: extend settings with agent knobs (`agent_enabled`, `agent_model`,
      `agent_auth_mode` chatgpt|api_key, api key env for fallback, codex home);
      update `.env.example`.
- [ ] CLI: add an argparse dispatch to `scufris` - default `serve`, plus
      `scufris login` (device-code) and `scufris chat "<prompt>"` (one turn) so
      the operator can verify the live path.
- [ ] Tests: `build_agent` returns `DisabledAgent` when off; `DisabledAgent`
      reply; `CodexAgent.chat` with the SDK boundary MOCKED (patch `AsyncCodex`)
      asserting it runs a turn and returns the text; a clear error when the SDK
      is absent; config parsing. App still boots with the agent disabled.
- [ ] Run `ruff check .`, `mypy .`, `pytest` green.

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
