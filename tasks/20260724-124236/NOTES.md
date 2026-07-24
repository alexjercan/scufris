# Backend-Agnostic Orchestrator Session Endpoints

## What Changed

- `fork_session`, `get_context`, `get_session_transcript`, and
  `delete_agent_session` now resolve the orchestrator's configured backend and
  call backend capabilities instead of reading Codex rollout files directly.
- `AgentBackend` gained `read_context` and async `delete_session` capabilities.
  Codex delegates to the rich rollout readers, Claude maps transcript status and
  unlinks its JSONL file, Opencode maps daemon message status and deletes through
  `OpencodeClient`, and Mock returns no-op values.
- `OpencodeClient` now exposes `delete_session`, returning `False` for 404s,
  other HTTP failures, and network failures.

## Why

The session list was already backend-agnostic, but the detail endpoints still
assumed Codex storage. That made a Claude or Opencode orchestrator able to list
sessions it could not open, inspect, fork, or delete. Routing through the backend
keeps session ownership and session operations under the same provider boundary.

## Tradeoffs

- `read_context` for Claude and Opencode intentionally reports only what their
  status readers expose. Their context window and Codex-only token axes remain 0
  instead of inventing unavailable data.
- `delete_session` is async in the protocol. This is a small shape change, but it
  avoids running an async `OpencodeClient` from inside a synchronous adapter while
  FastAPI is already in an event loop.

## Difficulties

- New app tests first failed because `Settings()` loaded the developer machine's
  real override store, which changed the backend under test. The fix was to pass
  an isolated `state_dir` for backend-sensitive app tests.
- The first opencode delete draft performed a blocking `httpx.delete` inside the
  backend. It was corrected so the opencode client owns daemon session deletion.

## Self-Reflection

I should have checked the exact ownership phrase in the task before writing the
first opencode delete test. The more useful test is at the client boundary, with
the backend test proving delegation through the configured backend.
