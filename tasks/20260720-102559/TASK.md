# Agent: steer the model to prefer the scufris MCP tools over raw shell

- STATUS: OPEN
- PRIORITY: 40
- TAGS: bug,agent,codex

## Goal

When asked host/tatr questions (e.g. "tell me details about this host"), the agent
runs raw shell commands (`uname`/`df`) instead of the curated scufris MCP tools
(`host_stats`, `disk_usage`, `list_processes`, `tatr_*`). Steer codex to PREFER the
scufris tools so those questions produce a tool call (and a tool chip in the UI).

Root cause: `agent.py:_exec_args` passes only the user prompt with no steering
instructions, and codex also has a read-only shell it reaches for by default.

## Notes

- Spike: tasks/20260720-102348/SPIKE.md.
- User feedback: asked "tell me details about this host" and it used bash instead
  of `host_stats`.
- Probe on the target host which codex lever actually works (base/experimental
  instructions via `-c`, an instructions file, or a prepended prompt preamble)
  WITHOUT dropping the read-only sandbox (see lessons `probe-runtime-on-target-host-early`,
  `codex-resume-rejects-sandbox`). A prompt preamble is the low-risk fallback.
- Verify LIVE that a "tell me about this host" turn emits a `host_stats` tool call.
  Applies to both `exec` and `app_server` backends.
