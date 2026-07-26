# the-den journal MCP tools (read/update habits, tasks, macros, weight via today/daily)

- STATUS: CLOSED
- PRIORITY: 40
- TAGS: feature,agent,mcp

## Flow State

- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Goal

Let the agent read and update the user's markdown journal ("the-den") in chat:
"what are today's tasks", "log 80kg", "check off gym", "add a note", "add a task
for tomorrow". Expose the-den as scufris MCP tools.

## Prerequisite: the unified `today` CLI (BUILT - contract VERIFIED 2026-07-26)

Design decided in tasks/20260720-140800/SPIKE.md. This task WRAPS the CLI; it does
not build it. The CLI now EXISTS: repo `~/personal/today`, binary `today` on PATH
(`~/.nix-profile/bin/today`). Verified contract against a temp den this session:

- Global flags come BEFORE the subcommand: `today --den <DIR> -N <offset> <sub>`.
  `--den <DIR>` selects the den (else `$DEN_PATH`, else `~/personal/the-den`);
  `-N/--offset <n>` picks the day (0=today, -1=yesterday, 1=tomorrow-as-a-day).
- read: `today --den D show --json` -> `{date,file,title,habits[],tasks[],
  tomorrow[],macros{protein,carbs,fat,calories},weight}`. NOTE: `show` does NOT
  include notes; notes read via `today --den D note list [--tag T] --json`.
  Each task is `{index,text,done}` (tomorrow items are `{index,text}`); each habit
  `{name,done}`; weight is a scalar-or-null.
- mutate (each `--json` returns the updated slice):
  `task add "text" [--tomorrow]`, `task done <index>` (today only, toggles),
  `task rm <index> [--tomorrow]`; `habit toggle <name>` (leading emoji optional);
  `weight <value>` (logs; no `--json` - prints "logged weight: N Kg");
  `macros add "what,protein,carbs,fat"`; `note add "text" [--tag T]`.
- error shape: a bad index / unknown habit / bad row prints a one-line message to
  stderr and exits non-zero (`_run` already folds stderr into the returned text).
  A nonexistent/unwritable den throws a Python traceback + exit 1 - so the wrapper
  MUST pre-check the den exists and short-circuit with a clean message.
- bare `today` opens `$EDITOR` (interactive - the tools NEVER call this; every tool
  passes an explicit subcommand).

## Design decisions (this task)

- den path: new `Settings.den_path: Path | None` (env `SCUFRIS_DEN_PATH`), default
  `None`. Injected into the ORCHESTRATOR scufris MCP env as `SCUFRIS_DEN_PATH` in
  `agent.scufris_mcp_server` (like `SCUFRIS_DISABLED_TOOLS`), read back in the MCP
  subprocess via a `_den_path()` helper (mirrors `_api_base()`/`_orch_session_id()`).
  Unset -> tools return a clear "journal not configured" error and never shell out;
  set-but-missing-dir -> clear error. This keeps scufris safe on a box without
  the-den. Default `None` (explicit opt-in), NOT the CLI's `~/personal/the-den`
  default, so scufris never silently creates a den.
- audience: journal tools are ORCHESTRATOR-only. The role scoping already gives the
  orchestrator every tool not in `_AGENT_ROLE_TOOLS`, and only the orchestrator env
  carries `SCUFRIS_DEN_PATH`, so a sub-agent never sees these. No `apply_role` change.
- surface: one fine-grained tool per operation (matches the existing one-tool-per-op
  pattern + strong "PREFER this over shell" descriptions). Nine tools, all prefixed
  `journal_` and routed through a shared `_journal(args)` helper that validates the
  den then calls `_run(["today", "--den", den, *args])`.
- read-only codex sandbox: the writes happen in the MCP server subprocess (not
  sandboxed), not via codex shell - the `_run` write path is confirmed by the temp
  -den tests.

## Steps

- [x] Add `den_path: Path | None = None` to `Settings` (config.py) with a doc
      comment (env `SCUFRIS_DEN_PATH`; unset = journal tools disabled).
- [x] Inject `SCUFRIS_DEN_PATH` into the orchestrator scufris MCP env in
      `agent.scufris_mcp_server` when `settings.den_path` is set (omit when None;
      never set for the agent role).
- [x] Add `_den_path()` + `_journal(args)` helpers to `mcp_server.py`: validate the
      den is configured and the dir exists, else return `error: ...`; otherwise
      `_run(["today", "--den", den, *args])`.
- [x] Add the nine `journal_*` tools with strong descriptions:
      `journal_show(offset=0)`, `journal_notes(tag="")`, `journal_add_task(text,
      tomorrow=False)`, `journal_complete_task(index)`, `journal_remove_task(index,
      tomorrow=False)`, `journal_toggle_habit(name)`, `journal_log_weight(value)`,
      `journal_add_macros(row)`, `journal_add_note(text, tag="")`.
- [x] Update `test_tools_registered` to include the nine journal tools.
- [x] Add MCP tests driving the REAL `today` CLI against a temp den (fixture copies
      `Templates/daily.md` so habits exist): show, add/complete/remove task (+
      tomorrow), toggle habit, log weight, add macros, add note + notes filter;
      plus den-unset and den-missing error paths (no shell-out).
- [x] Add an `agent.py` test: `scufris_mcp_server` injects `SCUFRIS_DEN_PATH` for the
      orchestrator when `den_path` is set, omits it when None, and never sets it for
      the agent role.
- [x] Document `SCUFRIS_DEN_PATH` in `.env.example` if that file lists the knobs.

## Definition of Done

1. The nine `journal_*` tools exist, are orchestrator-scoped, and each carries a
   description steering the model to use it over raw shell / file edits.
   (cmd: `nix develop -c python -m pytest tests/test_mcp_server.py -q`)
2. Reads and writes go through the real `today` CLI against a temp den and return
   its JSON; error paths (den unset, den missing, bad index) return a clean
   `error: ...` string, never a traceback.
   (cmd: `nix develop -c python -m pytest tests/test_mcp_server.py -q`)
3. `den_path` unset leaves scufris fully functional and the journal tools inert with
   a clear message; `SCUFRIS_DEN_PATH` is injected only into the orchestrator MCP env.
   (cmd: `nix develop -c python -m pytest tests/test_agent.py tests/test_mcp_server.py -q`)
4. Full local gate green: ruff, mypy, pytest.
   (cmd: `nix develop -c bash -c 'ruff check . && mypy . && python -m pytest -q'`)

## Implementation notes

- Deviation from plan (strengthening it): the `nix flake check` pytest sandbox runs
  with only `[virtualenv git cacert]` on PATH, so `today` is absent there (it is on
  PATH locally only because it leaks from the user nix profile). Rather than couple
  scufris's flake to the unpublished local `today` repo, the tests are TWO layers:
  (1) deterministic argv-contract + den-gating tests that stub `_run` and need no
  `today` (always green, incl. the sandbox); (2) real end-to-end tests driving the
  actual `today` CLI against a temp den, guarded by `skipif(shutil.which('today')
  is None)` so they run wherever it is installed and skip loudly in the sandbox.
  `nix flake check` is green; the real-CLI layer also ran green locally.
- Follow-up (separate task, NOT this branch): the DEPLOYED scufris service needs
  `today` on its PATH (like codex/claude/git in the nixos/home-manager module) for
  the journal tools to work off a dev box. File it as a new tatr task.
