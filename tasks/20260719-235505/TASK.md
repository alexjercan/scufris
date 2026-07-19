# Instrument agent, MCP tools and sessions with in-depth logs

- STATUS: OPEN
- PRIORITY: 35
- TAGS: feature, observability, agent, spike

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

## Notes

- Spike: tasks/20260719-235458/SPIKE.md.
- Depends on tatr 20260719-235504 (foundation) - needs `configure_logging`, the
  level plumbing, and the redaction helper.
- Verify at DEBUG a real (or fake-codex) turn shows the exec call, each tool call,
  the timing/usage, and that no secret/full-prompt leaks. Keep tests asserting the
  key log lines are emitted (caplog) rather than exact strings.
