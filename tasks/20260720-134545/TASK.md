# Settings: interactive 'try it' tool runner

- STATUS: OPEN
- PRIORITY: 20
- TAGS: feature,agent,ui

## Goal

An interactive "try it" runner on the Settings page: click a tool card -> a form
generated from its arg schema -> "Run" -> render the result, WITHOUT a chat turn.
Lets an operator debug a single tool in isolation ("does host_stats work right
now?").

## Notes

- Spike: tasks/20260720-134459/SPIKE.md (deferred out of task 122517 - it is
  interactive, not read-only).
- Needs a new backend endpoint that runs ONE scufris MCP tool by name with args
  and returns the result (bypassing codex/the agent) - a real capability, so mind
  the risk surface: a confirm step and possibly a gating setting.
- Arg form generated from the tool's `inputSchema.properties`; render the result
  (JSON/text). Escape everything.
- Do AFTER 122517 (which adds the health card + richer read-only tool cards).
