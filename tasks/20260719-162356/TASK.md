# Integrate Codex agent backend via openai-codex Python SDK (subscription auth)

- STATUS: OPEN
- PRIORITY: 20
- TAGS: feature,backlog,agent,llm

## Goal

Integrate OpenAI Codex as the Scufris agent backend via the official
`openai-codex` Python SDK, behind a small Scufris `Agent` interface, with
"Sign in with ChatGPT" (device-code) as the primary auth and an API key as a
fallback auth mode.

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
