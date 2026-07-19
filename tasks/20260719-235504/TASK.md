# Logging foundation: central config, debug mode, HTTP request logging

- STATUS: OPEN
- PRIORITY: 40
- TAGS: feature, observability, spike

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
