# A2b: claude (Claude Code headless) runner behind the AgentBackend interface

- STATUS: OPEN
- PRIORITY: 25
- TAGS: spike,agents,backend

## Goal

A second `AgentBackend` implementation: Claude Code in headless mode
(`claude -p --output-format stream-json`), behind the same interface as codex.
Normalize to the contract: session resume (`claude --resume`, its own on-disk
session store vs codex thread-id), MCP config (`--mcp-config` vs codex `-c`),
permission/sandbox model (`--permission-mode` / `--allowedTools` vs codex
`--sandbox`), and the status source (stream-json / session jsonl -> the same
`agent_status` shape). Building the second backend is what proves the A2
interface is not accidentally codex-shaped (decision 1). Includes its own
unattended probe.

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (decision 1 - codex first, claude right
  after).
- Depends on: 20260720-221935 (A2 - interface must exist first).
- Stepless direction-level task: run /plan before /work.
