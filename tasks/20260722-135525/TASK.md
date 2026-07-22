# Add opencode serve backend behind AgentBackend (adapt scufris-bot OpencodeClient) + settings/auth plumbing

- STATUS: OPEN
- PRIORITY: 10
- TAGS: spike, agent, backend

## Story

Add the third agent backend - `opencode` - behind the existing `AgentBackend`
protocol (`scufris/backends.py:104`), driving a running `opencode serve` daemon
(configured by 20260722-135520 to reach the host llama-server) over an adapted
async HTTP client. It must implement `stream`/`read_status`/`read_transcript`
like `CodexBackend`/`ClaudeBackend`, plug into every backend-branching site, and
carry the settings/auth plumbing a non-subscription local backend needs. v0
streaming uses opencode's SYNCHRONOUS `send_message` (block, emit the reply as a
text delta + done); live token streaming over `/event` is a deferred follow-up.

## Steps

- [ ] `scufris/enums.py`: add `OPENCODE = "opencode"` to `Backend`; add a
      no-subscription auth mode to `AuthMode` (`LOCAL = "local"`) for a backend
      whose endpoint needs no login.
- [ ] `scufris/opencode_client.py` (NEW): port/adapt the reference
      `scufris_server/opencode_client.py` - the exception taxonomy
      (`OpencodeNetworkError`/`ClientError`/`ServerError`/`Unavailable`/
      `StaleSessionError`), response models (`HealthResponse`, `Session`,
      `AssistantMessage`/`Part`, `TokenUsage`), request models
      (`SendMessageRequest`, `ModelRef`, `TextPartInput`), and methods
      `health` / `create_session` / `send_message` / `get_messages`
      (GET /session/:id/message, for status+transcript). Keep pydantic
      `extra="allow"`.
- [ ] `scufris/config.py`: register opencode in `_CANONICAL_BACKEND`,
      `available_backends` (unconditional - just needs the daemon), 
      `auth_mode_for_backend` (return `AuthMode.LOCAL`), `_BACKEND_LABELS`
      (`"Opencode"`), `_BACKEND_MODELS` (the host models),
      `default_model_for` (new `opencode_model`). Add `Settings` fields:
      `opencode_url` (default `http://127.0.0.1:4096`), `opencode_password`,
      `opencode_model` (default `gemma-4-26B-A4B-it`), `opencode_provider`
      (default `llamacpp`), `opencode_bin`, `opencode_auth_mode` (default
      `local`). scufris is a CLIENT of an already-running daemon (per the
      reference); launching/supervising `opencode serve` is out of scope here.
- [ ] `scufris/backends.py`: add `OpenCodeBackend` (name `"opencode"`):
      - `_OPENCODE_PERMISSION = {manual, edit, auto}` mapped to the per-tool
        permission mechanism recorded in 135520's NOTES (deny edit+bash / allow
        edit / allow all).
      - `stream(...)`: resolve `OpencodeClient` from settings; create a session
        (title from prompt) when no `session_id`, else reuse; build a
        `SendMessageRequest` with `ModelRef(opencode_provider, opencode_model)`
        and the permission mapping; `send_message`; yield a `StreamTool` per
        tool-call part, a `StreamTextDelta` of the reply text, then
        `StreamDone(reply, session_id)`. Map `OpencodeUnavailable`/errors to
        `StreamError`. Comment that live `/event` streaming is deferred.
      - `read_status(...)`: `get_messages` -> `BackendStatus` (turns, tool_calls,
        input/output tokens, last assistant text, updated_at).
      - `read_transcript(...)`: `get_messages` -> `list[TranscriptMessage]`.
      - register `"opencode"` in `get_backend`.
- [ ] `scufris/health.py`: add an `elif effective_backend == "opencode"` branch
      that probes the daemon via `OpencodeClient.health` (report version /
      unreachable) instead of a CLI `--version`.
- [ ] `scufris/app.py`: extend the model-key branch (~L960-967: pick
      `opencode_model` for an opencode agent) and re-check the codex-only gate
      (~L1240) so an opencode agent is handled, not mis-bucketed as codex.
- [ ] Tests: `tests/` unit tests for the client parsing (port the reference
      `tests/unit/test_opencode_client.py`), for the permission mapping, and for
      `read_status`/`read_transcript` parsing; plus ONE harness-level test that
      drives `OpenCodeBackend.stream` end to end against a faithful httpx mock
      (or the live daemon behind an opt-in marker) and asserts text+done events.
- [ ] `.env.example`: document `SCUFRIS_OPENCODE_URL`,
      `SCUFRIS_OPENCODE_PASSWORD`, `SCUFRIS_OPENCODE_MODEL`,
      `SCUFRIS_OPENCODE_PROVIDER`, `SCUFRIS_OPENCODE_AUTH_MODE`.
- [ ] Docs: update the README Agents section backend list (codex/claude ->
      + opencode) and write `tasks/20260722-135525/NOTES.md` (design record:
      sync-send v0, deferred /event streaming, permission mapping).

## Definition of Done

- `get_backend("opencode")` returns an `OpenCodeBackend` and
  `Backend.OPENCODE` is selectable (test:
  `test_get_backend_resolves_opencode`, `test_available_backends_includes_opencode`).
- `OpenCodeBackend.stream` yields a text delta then a `StreamDone` for a turn
  against a mocked daemon (test: `test_opencode_backend_streams_turn`).
- `read_status`/`read_transcript` parse a messages payload into
  `BackendStatus`/`TranscriptMessage`s (test:
  `test_opencode_read_status`, `test_opencode_read_transcript`).
- Permission modes map to opencode's per-tool config (test:
  `test_opencode_permission_mapping`).
- The client parsing tests pass (test: `test_opencode_client`).
- New settings are documented (cmd:
  `grep -n OPENCODE_URL .env.example`).
- `nix flake check` (ruff + mypy + pytest) passes (cmd: `nix flake check`).
- manual: with the real daemon up (from 135520), a turn through the backend
  returns a coherent reply from gemma-4-26B-A4B-it.

## Notes

- Spike: tasks/20260722-135404/SPIKE.md
- Depends on: 20260722-135520 (consumes its NOTES.md: provider/model id form +
  the per-tool permission mechanism; do not start before it lands).
- Reference infra: scufris-bot @ feature/opencode-v2 (opencode serve + llama.cpp)
  - port `scufris_server/opencode_client.py` and `tests/unit/test_opencode_client.py`.
- Test model: gemma-4-26B-A4B-it on host llama-server :11433.
- Backend-branching sites to touch (grepped 2026-07-22): config.py helpers,
  enums.py, backends.py `get_backend`, health.py (~L97/122/154), app.py
  (~L960-967, ~L1240). agent_store.py routes via `available_backends`/
  `canonical_backend`, so it picks up opencode automatically.
- v0: synchronous `send_message`; live `/event` token streaming deferred.
