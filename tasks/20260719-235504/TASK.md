# Logging foundation: central config, debug mode, HTTP request logging

- STATUS: CLOSED
- PRIORITY: 40
- TAGS: feature, observability, spike

## Implementation

- `config.py`: `log_level: str = "INFO"` (env SCUFRIS_LOG_LEVEL).
- `scufris/logsetup.py` (not `logging.py` - no stdlib shadow):
  `configure_logging(level, *, force=False)` (readable formatter, root stderr
  handler, sets scufris + uvicorn levels, first-wins idempotent, bad-level ->
  INFO); a `request_id` contextvar + `RequestIdFilter` + `new_request_id`; a
  `truncate` redaction helper.
- `cli.py`: `-v/--debug` accepted anywhere via a shared parent parser; the
  effective value read from argv by `_wants_debug` (argparse's parent-flag merge
  is unreliable); `main` resolves the level and `configure_logging(..., force=True)`
  before dispatch.
- `app.py`: an http middleware tags each request with an id and logs
  `METHOD path -> status in N ms` at DEBUG (5xx WARNING); `run_server` configures
  logging + logs startup + passes `log_config=None`/`log_level` to uvicorn.
- Tests: logsetup (levels, idempotent/force, bad-level, filter, truncate), CLI
  (`_wants_debug` all positions, accepted-anywhere, level selection), app
  (request is logged). Verified in-process: `DEBUG scufris.app [rid] GET
  /api/config -> 200 in 1.0ms`.

## Goal

The plumbing all in-depth logging builds on, with a dead-simple debug switch:

- `Settings.log_level: str = "INFO"` (env `SCUFRIS_LOG_LEVEL`).
- A `scufris/logging.py` `configure_logging(settings)` that installs a readable
  formatter (timestamp, level, logger, message), sets the `scufris` logger and
  uvicorn's loggers to the resolved level, and is idempotent.
- A `--debug` / `-v` CLI flag (global, before the subcommand) that overrides the
  level to DEBUG, wired through `cli.py` into every entrypoint (serve, login,
  chat, mcp-server) and `run_server`. So `scufris serve --debug` or
  `SCUFRIS_LOG_LEVEL=DEBUG scufris serve` both work.
- An HTTP request-logging middleware in `create_app`: method, path, status,
  duration_ms (and a request id via a `contextvars` filter, if adopted).

## Decisions (from /plan)

- Module name `scufris/logsetup.py` (NOT `logging.py`, which would shadow the
  stdlib `import logging` inside the package).
- `configure_logging(level, *, force=False)` is first-wins/idempotent: the CLI
  resolves the effective level and calls it with `force=True`; `run_server` calls
  it un-forced so a direct (non-CLI) launch still configures from
  `settings.log_level` but does not override the CLI's `--debug`.
- Request logging goes at DEBUG (so `--debug` shows every request without
  flooding the default INFO with the dashboard's 2s stats/processes polling);
  5xx responses log at WARNING regardless. A per-request id (uuid hex) rides a
  `contextvars` value + a `logging.Filter`, so every log line during a request
  carries `[<rid>]`.

## Steps

- [ ] `scufris/config.py`: `log_level: str = "INFO"` (env `SCUFRIS_LOG_LEVEL`).
- [ ] `scufris/logsetup.py`: `configure_logging(level, *, force=False)` (formatter
      `ts LEVEL name [rid] message`, root StreamHandler to stderr, sets scufris +
      uvicorn/uvicorn.access levels, idempotent); a `request_id` contextvar +
      `RequestIdFilter` + `new_request_id()`/`set_request_id()`; a `truncate(text,
      limit)` redaction helper for the instrumentation task to reuse.
- [ ] `scufris/cli.py`: a shared parent parser adds `-v/--debug` so it works both
      before and after the subcommand; `main()` resolves the level
      (`DEBUG` if `--debug` else `settings.log_level`) and `configure_logging(...,
      force=True)` before dispatching every command (serve, login, chat,
      mcp-server).
- [ ] `scufris/app.py`: an `@app.middleware("http")` that assigns a request id,
      times the request, and logs `METHOD path -> status in N ms` at DEBUG (5xx at
      WARNING). `run_server` calls `configure_logging(settings.log_level)`
      (un-forced) and passes `log_config=None` + `log_level` to uvicorn so it
      respects our config; drop the old `basicConfig`.
- [ ] Tests: `configure_logging` sets levels + is idempotent/force; the request
      middleware emits a log with the id (caplog + a TestClient request); the CLI
      `--debug` resolves DEBUG (parse + a configure spy). `ruff`/`mypy`/`pytest`
      green + a smoke: `scufris serve --debug` (or the parse path) selects DEBUG.

## Definition of Done

- `scufris serve --debug` and `SCUFRIS_LOG_LEVEL=DEBUG scufris serve` both start
  with verbose logging; a central `configure_logging` owns the format/levels;
  requests are logged (with a request id) at DEBUG and 5xx at WARNING; a
  `truncate` redaction helper exists for the next task. No stdlib-shadow; the
  non-CLI `run_server` path still configures. `ruff`/`mypy`/`pytest` green.

## Notes

- Spike: tasks/20260719-235458/SPIKE.md (chose stdlib logging, zero deps,
  terminal-readable; both a config knob and a `--debug` flag).
- Blocks tatr 20260719-235505 (instrumentation) - the loggers/levels must exist
  first for its DEBUG logs to surface.
- Keep the module named to avoid shadowing stdlib `logging` inside the package
  (import as `from . import applog` or name it `scufris/logconfig.py` - decide at
  /plan; a top-level `logging.py` in the package can shadow imports).
- Redaction helper (truncate prompt, redact secrets) can live here for the
  instrumentation task to reuse.
