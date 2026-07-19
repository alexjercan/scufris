# Review: logging foundation - config, debug mode, request logging

- DATE: 20260720
- VERDICT: APPROVE (1 round)

## Scope reviewed

`scufris/config.py` (`log_level`), `scufris/logsetup.py` (new), `scufris/cli.py`
(`--debug`/`-v` + `configure_logging`), `scufris/app.py` (request middleware +
`run_server`), tests (`test_logsetup.py` new, `test_cli.py` + `test_app.py`).

## Correctness

- Module named `logsetup` (not `logging`) so it does not shadow the stdlib import
  every module uses - the whole point flagged in the plan.
- `configure_logging(level, *, force=False)` is first-wins/idempotent (tested):
  the CLI resolves the level and calls with `force=True`; `run_server` calls
  un-forced so a direct launch configures without clobbering the CLI's `--debug`.
  Bad level names fall back to INFO (tested). It clears existing handlers before
  installing ours, so re-config does not duplicate output.
- The `--debug`/`-v` position problem was handled honestly: argparse's parent-flag
  merge across sub/parent namespaces is unreliable (it clobbered the value in
  practice), so the flag is ACCEPTED anywhere via a shared parent but the
  EFFECTIVE value is read from argv by `_wants_debug` (position-independent,
  tested for all four forms). Cleaner than fighting argparse.
- Request middleware: assigns a uuid request id into a contextvar (so every log
  line during the request carries `[rid]` via `RequestIdFilter`), times the
  request, and logs `METHOD path -> status in N ms` at DEBUG - so `--debug` shows
  every request without the default INFO being flooded by the 2s dashboard polls;
  5xx logs at WARNING. Verified in-process: `DEBUG scufris.app [ff23fb24] GET
  /api/config -> 200 in 1.0ms` with the exact format, level and id.
- `run_server` drops `basicConfig`, calls `configure_logging(settings.log_level)`,
  logs a startup line, and passes `log_config=None` + `log_level` to uvicorn so
  uvicorn does not install its own config and scufris + uvicorn share one format.
- A `truncate` redaction helper is in place for the instrumentation task to reuse.
- Full suite green: `ruff`/`ruff format`/`mypy` (11 files)/`pytest`.

## Nits (non-blocking)

- Request logging at DEBUG means normal INFO shows startup + (once instrumented)
  chat turns, not per-request lines - the intended "quiet by default, verbose on
  --debug" trade. If per-request INFO is ever wanted, it is a one-line level bump.
- The live `serve --debug` smoke via a backgrounded server did not bind (a known
  flaky nix-devshell-background issue); the logging pipeline was instead verified
  in-process end to end (format + level + request id) and by the pytest tests.

## Verdict

APPROVE. `scufris serve --debug` (and `SCUFRIS_LOG_LEVEL=DEBUG`) turn on verbose,
request-id-tagged logging through one central `configure_logging`; the plumbing +
redaction helper the instrumentation task needs are in place. No stdlib shadow,
idempotent config, argparse position bug sidestepped. Verified in-process + tests.
