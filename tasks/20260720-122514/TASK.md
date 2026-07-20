# the-den journal MCP tools (read/update habits, tasks, macros, weight via today/daily)

- STATUS: OPEN
- PRIORITY: 40
- TAGS: feature,agent,mcp

## Goal

Let the agent read and update the user's markdown journal ("the-den") in chat:
"what are today's tasks", "log 80kg", "check off gym", "add a note", "add a task
for tomorrow". Expose the-den as scufris MCP tools.

## Prerequisite: the unified `today` CLI (DESIGN LOCKED)

Design decided in tasks/20260720-140800/SPIKE.md (user questionnaire). This task
WRAPS that CLI; it does not build it. The CLI is a NEW external repo
`~/personal/today` (Python, structured like tatr, own flake+overlay) that merges
the two bash scripts and replaces `today.nix`/`daily.nix` in nix.dotfiles.

Confirmed CLI contract this task wraps (command name `today`, subcommands, `--json`):
- read: `today show [-N offset] --json` -> {date,file,title,habits,tasks,tomorrow,
  macros,weight,notes}; `today path [-N]`.
- mutate (each `--json` returns the updated slice): `today task add "text"` /
  `today task done <idx>` / `today task rm <idx>` (+ tomorrow variant);
  `today habit toggle <name>`; `today weight <value>`;
  `today macros add "what,protein,carbs,fat"`; `today note add "text" [--tag]`.
- bare `today` opens `$EDITOR` (interactive - the agent NEVER calls this).

## Sequencing

- Build the `~/personal/today` CLI FIRST (its own repo/flow), then this task wraps
  its subcommands. The "start against current daily and migrate" interim is now
  OFF (the user chose to build the CLI first, to avoid rework).

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
