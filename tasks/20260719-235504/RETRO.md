# Retro: logging foundation

- DATE: 20260720
- VERDICT: APPROVE (1 review round)

## What went well

- Small, well-scoped plumbing task; the spike's decisions (stdlib, zero deps,
  `--debug` + `log_level`, `logsetup` not `logging`) held, so it was a mechanical
  build. The `truncate` helper + request-id filter are ready for the
  instrumentation task to consume.
- Verifying the actual formatted output in-process ("`DEBUG scufris.app [ff23fb24]
  GET /api/config -> 200 in 1.0ms`") caught nothing wrong but proved the whole
  pipeline (format, level, contextvar request id) end to end, which a caplog
  message-only assertion would have missed.
- The "quiet by default, verbose on --debug" split (request logs at DEBUG) keeps
  the default INFO from drowning in the dashboard's 2s polling - the right call
  for a tool an operator watches in a terminal.

## What went wrong / friction

- The argparse parent-flag position bug bit exactly as the "unreliable" hunch
  suspected: with `-v/--debug` on a shared parent applied to BOTH the top parser
  and the subparsers, `serve --debug` and `--debug serve` disagreed (the subparser
  default clobbered the parent's value). Tried `default=argparse.SUPPRESS` +
  `set_defaults` - still wrong for the pre-command position. The clean fix was to
  stop trusting argparse's merge and read the flag straight from argv
  (`_wants_debug`), keeping the parser only for acceptance/validation.
- The live `serve --debug` smoke (backgrounded server in the nix dev shell) did
  not bind again - the same flaky pattern from earlier tasks. Verified in-process
  instead; the pipeline is what mattered.

## Lessons

- `argparse-global-flag-read-from-argv` (CLI): a global flag that must work BOTH
  before and after a subcommand is unreliable via `parents=` on top + subparsers
  (the subparser default clobbers the value; `SUPPRESS`/`set_defaults` do not
  fully fix it). Add it to a shared parent only so argparse ACCEPTS it anywhere,
  then read the effective value straight from argv (`"--debug" in argv`), not from
  `args.<dest>`. 20260719-235504.

## Follow-ups

- Next: instrument agent/MCP/sessions (tatr 20260719-235505) using
  `configure_logging` + `truncate` from here.
- If per-request INFO logging is ever wanted (not just DEBUG), it is a one-line
  level bump in the middleware.
