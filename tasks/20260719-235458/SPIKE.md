# Spike: in-depth logging + easy debug mode for scufris

- DATE: 20260719-235458
- STATUS: RECOMMENDED
- TAGS: spike, observability

## Question

scufris has almost no logging. How should we add in-depth, operator-readable
logging - agent tool calls, codex/CLI subprocess invocations, API requests,
session operations - and a dead-simple way to start it in a verbose debug mode?
A good answer picks a logging approach (deps? structured?), names WHERE each
category is logged, and defines the debug switch.

## Context

Today logging is effectively nothing: only `scufris/app.py` has a logger
(`logging.getLogger(__name__)`), used for ONE warning (missing web dist).
`run_server` calls `logging.basicConfig(level=logging.INFO)` and that is the
whole setup. There is no log-level/debug config knob, no CLI flag. The parts the
user wants visibility into are silent:

- **agent** (`scufris/agent.py`): drives `codex exec` as a subprocess (the CLI
  call), parses `mcp_tool_call` events (the agent tool calls) and token usage -
  none logged. Both the blocking `_run_codex_exec` and the streaming
  `_stream_codex_exec` paths.
- **MCP server** (`scufris/mcp_server.py`): `_run(...)` shells out to `df`, `tatr`,
  etc. (the "CLI calls") - silent. NB this runs as a SEPARATE process spawned by
  codex, so its logs are not in the server's stream (see Open questions).
- **sessions** (`scufris/sessions.py`): list/read/delete rollout files - silent.
- **API requests**: only uvicorn's default access log; no per-request scufris log
  with timing/status.
- `scufris/cli.py` uses `print()` for the chat reply.

The uv2nix build tolerates pure-Python deps (the codex-binary problem was a
NATIVE wheel; a pure-Python logging lib is fine), so a dep is not off the table.

## Options considered

### Logging library

- **Python stdlib `logging` + a central `configure_logging(settings)` (RECOMMENDED).**
  A single setup function installs a readable formatter (`ts level logger:
  message`), sets the `scufris` logger + uvicorn's (`uvicorn`, `uvicorn.access`)
  to the chosen level, and every module uses `logging.getLogger(__name__)`. Rich
  context goes in the message as `key=value` (e.g. `codex exec resume model=gpt-5.5
  tools=3 -> exit=0 in 4.2s`). Pros: zero deps, standard, integrates with
  uvicorn's own logging, an operator reads it in a terminal. Cons: not
  machine-structured (JSON); request-id propagation is manual (a middleware sets
  it and includes it in the message, or a `logging.Filter` + contextvar).
- **structlog (rejected for now).** Structured key-value / JSON events, contextvar
  binding for a request id that auto-attaches to every log in that request. Pure
  Python, so it BUILDS in uv2nix. Pros: machine-parseable, clean per-event
  context. Cons: a dep + a learning curve for a single-host tool whose operator
  reads logs by eye; JSON is worse than aligned text in a terminal. Revisit if
  logs ever need shipping to a collector.
- **loguru (rejected).** Nicer API, but it fights stdlib/uvicorn logging
  integration and is another dep for marginal gain here.

### Debug switch (how you turn it on)

- **RECOMMENDED: a `log_level` config knob PLUS a `--debug`/`-v` CLI flag.**
  `Settings.log_level: str = "INFO"` (env `SCUFRIS_LOG_LEVEL`), and
  `scufris serve --debug` (also `scufris -v ...`) overrides it to `DEBUG`. So the
  easy paths are `scufris serve --debug` or `SCUFRIS_LOG_LEVEL=DEBUG scufris serve`.
  `configure_logging` reads the resolved level. A global `--debug` before the
  subcommand covers `login`/`chat`/`mcp-server` too.
- Env-only (rejected as the sole mechanism): a flag is the "easy" the user asked
  for; keep both.

### What to log, and at which level

- **INFO** (always on): server start (host/port, agent on/off), each chat turn
  (session, #tools, tokens, duration), each API request (method path -> status in
  Nms), each MCP tool call (server.tool -> status), session delete/switch/fork.
- **DEBUG** (`--debug`): the full `codex exec` argv (minus/red-acting the prompt
  body, or truncated), each `_run` command + duration + exit + output length,
  each streamed `--json` event line, session-scan details, request bodies size.
- Redaction: never log the OpenAI API key or full prompt text at INFO; truncate
  prompts and redact secrets even at DEBUG.

## Recommendation

Stdlib `logging` with a central `configure_logging`, a `log_level` setting, and a
`--debug` CLI flag - zero deps, terminal-readable, uvicorn-integrated. Split into
two flows:

1. **Foundation (tatr 20260719-235504)** - the plumbing everything else uses:
   `Settings.log_level`; a `scufris/logging.py` `configure_logging(settings)`
   (formatter, scufris + uvicorn levels, optional request-id contextvar filter);
   a `--debug`/`-v` flag wired through `cli.py` into every entrypoint (serve,
   login, chat, mcp-server) and `run_server`; and an HTTP request-logging
   middleware in `create_app` (method, path, status, duration_ms, request id).
2. **Instrumentation (tatr 20260719-235505)** - add the actual logs, using the
   foundation: `agent.py` (codex exec invocation + timing + exit, each tool call,
   token usage - both runners), `mcp_server.py` (`_run` command/duration/exit, and
   configure its own logging in `main()` since it is a separate process), and
   `sessions.py` (list/read/delete counts). With redaction/truncation of prompts
   and secrets.

## Open questions

- **The MCP server is a separate process.** Its logs go to its own stderr, which
  codex may swallow. For debug visibility, `mcp_server.main()` should
  `configure_logging` from env (`SCUFRIS_LOG_LEVEL`) and log to stderr; optionally
  a `SCUFRIS_MCP_LOG_FILE` to tee to a file. Decide the default at that task's
  /plan (stderr-only is the simplest honest default).
- **Request-id propagation**: a `contextvars` id set by the middleware and pulled
  into every log line via a `logging.Filter` is the clean stdlib way - worth it,
  or is method+path enough? Lean to including it (cheap, big debugging win) but
  confirm at /plan.
- **codex's own verbosity** (`RUST_LOG`/`--verbose`) is codex's concern; out of
  scope beyond logging that we invoked it.

## Next steps

Direction-level tasks seeded (for `/plan` to break into steps):

- tatr 20260719-235504: Logging foundation - central config, debug mode, HTTP request logging
- tatr 20260719-235505: Instrument agent, MCP tools and sessions with in-depth logs

## Fix record

(Appended by each implementing task as it lands.)
