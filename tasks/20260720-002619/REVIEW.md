# Review: agent backend via codex app-server (streaming deltas)

- DATE: 20260720
- VERDICT: APPROVE (1 round)

## Scope reviewed

`scufris/config.py` (`agent_backend`), `scufris/agent.py` (`StreamTextDelta`/
`StreamReasoningDelta`, `_appserver_event`, `_appserver_usage`, `_appserver_call`,
`_stream_app_server`, `build_agent` selection), `tests/test_agent.py`.

## Correctness

- Live-verified END TO END with a REAL codex app-server turn through the SSE
  endpoint (`agent_backend=app_server`): 14 `text_delta` frames streamed
  `1, 2, 3, 4, 5.` token-by-token, then a `done` with the same assembled reply.
  content-type `text/event-stream`. This is the token-by-token the user asked for,
  proven on the real protocol - not just fakes.
- The protocol was de-risked by a probe first (recorded in the task): initialize
  -> thread/start|resume -> turn/start, then notifications
  (`item/agentMessage/delta`, `item/reasoning/textDelta`, `item/completed` tools,
  `thread/tokenUsage/updated`, `turn/completed`). `_stream_app_server` implements
  exactly that with a wall-clock deadline and a `finally` that kills the process
  (client disconnect), mirroring the exec streamer.
- `_appserver_event` is a pure, unit-tested mapper (text delta, reasoning delta,
  tool item -> StreamTool, None otherwise); `_appserver_usage` reads the camelCase
  totals. The runner accumulates text + tools and yields `StreamDone{reply,
  session_id=thread_id}` so multi-turn continues via `thread/resume`.
- Additive + config-gated: `agent_backend` defaults to `exec`, so the shipped
  default is unchanged; `build_agent` selects the stream runner by the flag (unless
  a test injects one), and `chat()`/CLI/fork always stay on `_run_codex_exec`. The
  SSE endpoint forwards the new event kinds unchanged (`model_dump_json`).
- The fake-codex JSON-RPC integration test drives the real `_stream_app_server`
  handshake + delta emission (asserts `["Hel","lo"]` -> `"Hello"`, session `t-1`,
  usage), and the backend-selection test pins `build_agent`.
- Full suite green: `ruff`/`ruff format`/`mypy` (11 files)/`pytest`.

## Nits (non-blocking)

- Tool-event fidelity: the `item/completed`-with-a-tool mapping is best-effort
  (the probe's simple turns did not force a tool call, so exact field names are
  inferred defensively: server/tool/name/status with fallbacks). A real
  tool-calling app_server turn should be eyeballed when the UI lands; text +
  reasoning + done are fully confirmed.
- The FRONTEND is unchanged here (next task): with `app_server` on, the current
  SSE consumer would treat `text_delta` as an unknown kind. That is the UI task
  (20260720-002621); the default `exec` backend keeps the shipped UI correct.
- EXPERIMENTAL protocol: pinned behind the flag with `exec` as the proven
  fallback, and the schema can be re-inspected on codex upgrades - the accepted
  trade for token-by-token.

## Verdict

APPROVE. A codex app-server streaming backend delivers real token-by-token text
(+ reasoning + tool events + usage) behind the Agent seam, config-gated so exec
stays the default/fallback. Probe-first de-risking, live-verified on the real
protocol, additive, and tested. Ready for the UI task to render it.
