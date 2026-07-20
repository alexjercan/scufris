# the-den journal MCP tools (read/update habits, tasks, macros, weight via today/daily)

- STATUS: OPEN
- PRIORITY: 40
- TAGS: feature,agent,mcp

## Goal

Let the agent read and update the user's markdown journal ("the-den") in chat:
"what are today's tasks", "log 80kg", "check off gym", "add a note", "add a task
for tomorrow". Expose the-den as scufris MCP tools.

## Prerequisite / re-scope (user, 20260720)

The user wants a better FOUNDATION than the current two bash scripts:
- The existing `today` (open/create today's entry) and `daily` (read/mutate) are
  TWO separate bash scripts. The user wants them merged into a SINGLE, unified,
  agentic-friendly command.
- Preferred: build a NEW dedicated project under `~/personal` (the way `tatr` is
  its own project) that implements this unified journal CLI in PYTHON, targets
  the-den, and REPLACES the `today.sh`/`daily.sh` bash scripts in the nix config
  (`nix.dotfiles/home/modules/scripts/{today,daily}.nix`).
- scufris then wraps THAT unified CLI as MCP tools.

That new CLI is SEPARATE work in its OWN repo (not scufris) and should get its
OWN `/spike` (unified-command design, agentic JSON in/out API, nix packaging,
replacing the scripts). It is recorded here as the dependency, not built from this
scufris task.

## Sequencing

- Preferred: build the unified `den` CLI first, then this task wraps it (clean).
- Acceptable interim: this task can START against the CURRENT `daily`/`today` and
  migrate to the unified CLI's interface later - but confirm with the user first,
  since they may want the CLI done first to avoid rework.

## Notes (scufris side)

- User: today/daily target `the-den` (`/home/alex/personal/the-den`), a journal of
  `.md` files (Daily/, Notes/, Templates/, tasks/).
- Current CLI surface (until the unified CLI lands), all non-interactive:
  `daily [DEN] --json` -> `{date,file,title,habits,tasks,tomorrow,macros,weight}`;
  mutations `--toggle-habit`, `--task-entry/-remove`, `--toggle-task`,
  `--task-tomorrow-entry/-remove`, `--weight-entry`, `--macros-entry`,
  `--notes-entry`, `--offset N`; `today -p` prints today's path.
- Add a `den_path` config knob (the dotfiles already set `user.journal.den_path`);
  the tools must gracefully no-op / report clearly when it is unset or missing, so
  scufris stays safe on a box without the-den. Gate under `agent_tools_enabled`.
- Mind the read-only codex sandbox: these tools mutate files under the-den via the
  journal CLI in the MCP server process (which is not sandboxed), NOT via codex
  shell - confirm the write path works from the MCP server.
- Follow the existing `mcp_server.py` tool patterns + tests (real CLI against a
  temp den). Reinforce with strong tool descriptions (see the tool-steering task).
