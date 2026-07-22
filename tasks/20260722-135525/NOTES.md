# NOTES: the opencode backend (design record)

Implements the third `AgentBackend` (`scufris/backends.py`), driving a running
`opencode serve` daemon aimed at a self-hosted llama.cpp server. Direction:
`tasks/20260722-135404/SPIKE.md`; the live API contract it builds on:
`tasks/20260722-135520/NOTES.md`.

## Shape

- `scufris/opencode_client.py` - async httpx client for the daemon, adapted from
  the `scufris-bot` reference, trimmed to `health`, `create_session`,
  `send_message` (synchronous turn) and `get_messages`. Error taxonomy:
  network / client(4xx) / server(5xx) / unavailable(health) / stale-session(404).
  Response models use `extra="allow"` so opencode can add fields freely.
- `scufris/backends.py::OpenCodeBackend` - `stream` / `read_status` /
  `read_transcript` behind the protocol, plus `get_backend("opencode")`.

## Decisions

- **Client of a daemon, not a subprocess.** opencode's headless surface is the
  `opencode serve` HTTP daemon (codex `app_server` shape), so scufris talks to it
  over HTTP rather than spawning a CLI per turn like codex/claude. Launching /
  supervising the daemon is out of scope (the operator runs it; `opencode_url`
  points at it). This is the deliberate seam-stress: a backend that is neither
  stdio-subprocess (claude) nor JSON-RPC-over-stdio (codex).
- **Synchronous turn (v0).** `stream()` calls the blocking `send_message` and
  emits the whole reply as one `StreamTextDelta` + `StreamDone` (plus a
  `StreamTool` per tool part). Live token streaming over the daemon's `/event`
  SSE bus is a deliberate follow-up, not shipped here - the goal is a working
  backend behind the seam, and the local model's tool-calling is weak anyway
  (135520). Filed nothing blocks upgrading later; the event vocabulary is the
  same `StreamEvent`.
- **Permission mode -> `tools` map.** manual|edit|auto map to a per-request
  `tools` boolean map that DISABLES tools (`_OPENCODE_PERMISSION`): manual turns
  off edit/write/patch/bash (read-only), edit turns off bash, auto sends nothing
  (all on). Chosen over named config-agents because a headless server has no one
  to answer opencode's `ask` approvals, so availability (a disabled tool cannot
  be called) is the safe, per-request lever. The message endpoint accepts both
  `agent` and `tools` (verified via the daemon OpenAPI, 135520); `tools` needs no
  shared config state.
- **Sync read path.** `read_status`/`read_transcript` are synchronous protocol
  methods called from the async app, so `_read_messages` uses a blocking
  `httpx.get` (an `asyncio.run` would raise inside the running loop). This
  mirrors codex/claude reading their session files synchronously at the same
  call site. Any failure -> None/[] (never crash a snapshot).
- **Stale session recovery.** A 404 on `send_message` (a deleted id, or a
  cross-backend id after a backend switch) is caught and the turn retried once
  against a freshly created session - the same "start fresh instead of failing
  the turn" behaviour claude's backend has for unresumable ids.

## Wiring touched

`enums.py` (`Backend.OPENCODE`, `AuthMode.LOCAL`); `config.py` (settings
`opencode_url/password/model/provider` + `agent_opencode_auth_mode`, and the
`_CANONICAL_BACKEND` / `available_backends` / `auth_mode_for_backend` /
`_BACKEND_LABELS` / `_BACKEND_MODELS` / `default_model_for` registrations);
`backends.py` (the class + factory); `health.py` (a daemon-health probe branch);
`app.py` (orchestrator model-key selection). `agent_store` needs no change - it
routes through `available_backends`/`canonical_backend`. The usage/memory/account
panels stay None for opencode (no rollout reader), same as claude.

## Not done / follow-ups

- Live `/event` token streaming (v0 is synchronous).
- Image attachments (FilePartInput), like the claude path's deferral.
- A `scufris login` branch: opencode needs no login (auth mode `local`); the CLI
  login flow is codex/claude only and simply does not apply.
