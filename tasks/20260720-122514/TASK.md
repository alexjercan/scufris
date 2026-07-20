# the-den journal MCP tools (read/update habits, tasks, macros, weight via today/daily)

- STATUS: OPEN
- PRIORITY: 40
- TAGS: feature,agent,mcp

## Goal

Let the agent read and update the user's markdown journal ("the-den") in chat:
"what are today's tasks", "log 80kg", "check off gym", "add a note", "add a task
for tomorrow". Add scufris MCP tools wrapping the existing `daily`/`today` CLIs,
which are already non-interactive with JSON I/O.

## Notes

- Spike: tasks/20260720-122301/SPIKE.md.
- User: today/daily target `the-den` (`/home/alex/personal/the-den`), a journal of
  `.md` files (Daily/, Notes/, Templates/, tasks/).
- CLI surface to wrap (all non-interactive):
  `daily [DEN] --json` -> `{date,file,title,habits,tasks,tomorrow,macros,weight}`;
  mutations `--toggle-habit`, `--task-entry/-remove`, `--toggle-task`,
  `--task-tomorrow-entry/-remove`, `--weight-entry`, `--macros-entry`,
  `--notes-entry`, `--offset N`; `today -p` prints today's path.
- Add a `den_path` config knob (the dotfiles already set `user.journal.den_path`);
  the tools must gracefully no-op / report clearly when it is unset or missing, so
  scufris stays safe on a box without the-den. Gate under `agent_tools_enabled`.
- Mind the read-only codex sandbox: these tools mutate files under the-den via the
  `daily` binary in the MCP server process (which is not sandboxed), NOT via codex
  shell - confirm the write path works from the MCP server.
- Follow the existing `mcp_server.py` tool patterns + tests (real CLI against a
  temp den). Reinforce with strong tool descriptions (see the tool-steering task).
