# Spike: the unified "today" journal CLI (merges today+daily, targets the-den)

- DATE: 20260720-140800
- STATUS: RECOMMENDED
- TAGS: spike, agent, den

## Question

The scufris den MCP tools (task 122514) should expose the-den journal in chat, but
the current tooling is ~1200 lines of bash+jq maintained in nix.dotfiles as TWO
commands. What should the unified, agentic-friendly journal CLI be, so scufris can
wrap it cleanly? This spike locks the design (decided with the user via a
questionnaire) so the CLI can be built in its own repo and then wrapped.

## Context

`the-den` (`/home/alex/personal/the-den`) is a markdown journal: `Daily/YYYY-MM-DD-
Weekday.md` (sections: Habits `- [ ]`, Macros CSV, Notes, Tasks/Tomorrow), plus
`Notes/`, `Templates/daily.md`, `tasks/`. It is DATA only. The tools live in
nix.dotfiles as `home/modules/scripts/{today,daily}.nix`:
- `today` (bash): opens/creates today's entry in `$EDITOR`; `-t` template, `-c`
  create+print-path, `-p` print-path; a templator fills `{{title}}` and carries
  yesterday's "Tomorrow" list forward.
- `daily` (bash+jq, 1159 lines): non-interactive read/mutate - `--json`,
  `--toggle-habit`, `--task-entry/-remove`, `--toggle-task`, `--task-tomorrow-*`,
  `--weight-entry`, `--macros-entry`, `--notes-entry`, `-n <tag>`, `-N <offset>`.

`tatr` (`~/personal/tatr`) is the structural reference: its own repo + `flake.nix`
(exports an overlay consumed by nix.dotfiles) + README + AGENTS.md + `tasks/`. tatr
is C; the user wants THIS one in Python.

## Decisions (user, via questionnaire 20260720-140800)

1. **Repo**: a NEW standalone `~/personal/today` repo (the project is named
   `today`), versioned independently, exporting a nix overlay that REPLACES
   `today.nix` + `daily.nix` in nix.dotfiles. the-den stays pure data.
2. **CLI surface**: one command `today` with SUBCOMMANDS (not flags, not two
   binaries). e.g. `today show --json`, `today task add "..."`, `today task done 2`,
   `today habit toggle Gym`, `today weight 80`, `today note add "..."`.
3. **Editor**: bare `today` (no args) opens today's `.md` in `$EDITOR` (create +
   template + carry-forward, like the current `today`); every DATA operation is a
   non-interactive subcommand (so agents only ever call subcommands, never the
   editor).
4. **Scope**: parity + a few improvements (port everything today+daily do, plus
   small wins: note tags/search, habit streaks, weight trend).

## Recommendation

Build `~/personal/today` as a Python CLI, structured like tatr (own flake + README
+ AGENTS.md + tasks/ + tests), with this shape:

- **Entry / editor**: `today [-N <offset>] [--den PATH]` with no subcommand ->
  create-if-missing (apply `Templates/daily.md`, fill `{{title}}`, carry the
  "Tomorrow" section from the previous entry) and open `$EDITOR`. `today path [-N]`
  prints the path without opening (replaces `today -p`); `today create [-N]`
  creates + prints path without opening (replaces `today -c`).
- **Read**: `today show [-N] [--json]` -> the day's habits/tasks/tomorrow/macros/
  weight/notes; `--json` emits one machine-readable object (the agentic contract -
  mirror the current `daily --json` shape: {date,file,title,habits,tasks,tomorrow,
  macros,weight,notes}).
- **Mutations** (each accepts `-N` for other days; each supports `--json` to return
  the updated slice for agents):
  - tasks: `today task add "text"`, `today task done <idx>`, `today task rm <idx>`,
    and a `--tomorrow` variant (or `today tomorrow add/rm`).
  - habits: `today habit toggle <name>`, `today habit list`.
  - weight: `today weight <value>` (log), `today weight` (show + trend improvement).
  - macros: `today macros add "what,protein,carbs,fat"`.
  - notes: `today note add "text" [--tag TAG]`, `today note list [--tag TAG]`
    (tags + search = the notes improvement).
- **Config**: den path from `--den` / `$DEN_PATH` (or the-den default); the CLI is
  the ONLY writer of the format, so scufris never parses markdown itself.
- **Contract for agents**: read commands default human-readable, `--json` for
  machines; mutations are idempotent-ish and return the updated state with `--json`;
  non-zero exit + a clear stderr message on error (never a half-write).
- **Packaging**: `flake.nix` exporting `overlays.default` (like tatr) so
  nix.dotfiles swaps `programs.today`/`daily` for this package and deletes the two
  `*.nix` scripts; keep the markdown format 100% backward compatible with existing
  `Daily/*.md`.

Why this shape: subcommands + `--json` is exactly what an MCP tool wants to wrap
(122514 maps ~1:1 to subcommands); keeping the bare-command editor preserves the
human "jump into today" flow without any interactive surface for the agent; a
separate repo + overlay matches the established tatr pattern and lets nix.dotfiles
retire the bash.

## Open questions

- **Improvement details** (habit streaks / weight trend / note search) - exact
  output shape to settle during that repo's own `/plan`; not blocking the parity core.
- **Format edge cases** - the templator's "carry Tomorrow forward" and the exact
  Macros CSV / weight storage need to be read out of the current bash + real
  `Daily/*.md` before porting (behavioral parity by example, per
  `capture-real-cli-output-for-parser-tests`).
- **scufris den_path knob** - 122514 adds `settings.den_path`, passed to the CLI as
  `--den`; no-op safely when unset.

## Next steps

This CLI is EXTERNAL work (its own `~/personal/today` repo), so it does not get
scufris tatr tasks. The build sequence:

1. Bootstrap `~/personal/today` (Python project, flake + overlay + README + AGENTS.md
   + tasks/, structured like tatr) and build the CLI to the shape above - its own
   spike/plan/flow lives in THAT repo.
2. Point nix.dotfiles at it (overlay) and delete `today.nix`/`daily.nix`.
3. scufris tatr 20260720-122514: wrap the `today` subcommands as MCP tools (updated
   with this confirmed contract).

## Fix record

(Appended as the pieces land.)
