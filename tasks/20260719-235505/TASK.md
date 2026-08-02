# Instrument agent, MCP tools and sessions with in-depth logs

- PRIORITY: 35
- TAGS: feature, observability, agent, spike
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Implementation

- `agent.py`: module logger + `_log_tool_call`/`_log_usage` helpers; both runners
  log the exec (DEBUG: mode/model/truncated prompt), each tool call (INFO), token
  usage (INFO), a timing summary (INFO), and the streaming runner logs each raw
  `codex json:` line (DEBUG); timeout WARNING, failure ERROR. Prompt truncated,
  API key never logged.
- `mcp_server.py`: `_run` logs command/exit/bytes/duration (DEBUG) + failures
  (INFO); `main()` configures logging from `SCUFRIS_LOG_LEVEL` (separate process).
- `sessions.py`: list (DEBUG count) + delete (INFO).
- Tests (caplog, level-targeted): run/stream exec log the exec + tool + usage +
  prompt-truncation; `_run` logs; `mcp_server.main` configures + runs; delete
  logs. Live-verified the full DEBUG trace via fake-codex. `ruff`/`mypy`/`pytest`
  green.

## Goal

Add the actual in-depth logs the operator wants, using the logging foundation:

- **agent.py** (both `_run_codex_exec` and `_stream_codex_exec`): log the
  `codex exec` invocation (resume?, model, tool count) at INFO with timing + exit;
  the full argv (prompt truncated/redacted) at DEBUG; each `mcp_tool_call`
  (server.tool -> status) at INFO; token usage at INFO; each streamed `--json`
  event line at DEBUG.
- **mcp_server.py**: log each `_run` CLI call (command, duration, exit, output
  length) - DEBUG for the command, INFO for a failure; and `configure_logging`
  from env in `main()` since it runs as a SEPARATE process (stderr; a
  `SCUFRIS_MCP_LOG_FILE` tee is optional).
- **sessions.py**: log list/read/delete/fork operations with counts at
  INFO/DEBUG.
- Redact the OpenAI key and truncate prompt text everywhere (reuse the
  foundation's redaction helper).

## Steps

- [x] `agent.py`: module logger; in `_run_codex_exec` and `_stream_codex_exec` log
      the exec at DEBUG (mode, model, prompt via `truncate`) + an INFO summary
      (mode -> exit/tools/timing); each `mcp_tool_call` at INFO (`tool
      server.tool -> status`); token usage at INFO; timeout/failure at
      WARNING/ERROR. Never log the API key; truncate the prompt.
- [x] `mcp_server.py`: log each `_run` (DEBUG: command, exit, bytes, duration;
      INFO on failure); `configure_logging(SCUFRIS_LOG_LEVEL)` in `main()` since it
      is a separate process (stderr).
- [x] `sessions.py`: module logger; DEBUG counts for list/read, INFO for delete.
- [x] Tests (caplog, level-targeted): the stream/run exec logs the exec + a tool
      line + usage; a long prompt is truncated (full text NOT in the logs);
      `_run` logs; `mcp_server.main` configures logging. `ruff`/`mypy`/`pytest`
      green; a DEBUG fake-codex turn shows the full trace.

## Definition of Done

- At DEBUG, a turn shows: the codex exec invocation (mode/model/truncated prompt),
  each tool call, token usage, and timing; MCP `_run` CLI calls; session ops. No
  API key or full prompt leaks. INFO stays useful (exec summary, tool calls,
  usage) without DEBUG noise. `ruff`/`mypy`/`pytest` green; fake-codex verified.

## Notes

- Spike: tasks/20260719-235458/SPIKE.md.
- Depends on tatr 20260719-235504 (foundation) - needs `configure_logging`, the
  level plumbing, and the redaction helper.
- Verify at DEBUG a real (or fake-codex) turn shows the exec call, each tool call,
  the timing/usage, and that no secret/full-prompt leaks. Keep tests asserting the
  key log lines are emitted (caplog) rather than exact strings.

> Hygiene pass 20260720-220123: the step box(es) above were ticked
> retroactively to clear a `closed-unchecked` finding. `scufris/sessions.py` has the module logger with DEBUG list/read + INFO delete counts (verified in code at sessions.py:64,276,314); the box was left unticked.
