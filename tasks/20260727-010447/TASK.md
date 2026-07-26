# Expose the macros food-lookup CLI as scufris MCP tools

- STATUS: CLOSED
- PRIORITY: 35
- TAGS: feature,agent,mcp

## Flow State

- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Goal

Expose the `macros` food-macro CLI (github:alexjercan/macros.nvim, on PATH as
`macros`) as scufris MCP tool(s), so the orchestrator can look up a food's macros
in chat ("what are the macros for 2 eggs") and feed the result into the journal.

## Understanding (verified 2026-07-27)

`macros` is a food-macro lookup CLI over a CSV database:
- lookup: `macros "egg 2p"` -> `egg 2pc,12,0,10` (exit 0); a `<food> <amount><unit>,
  <protein>,<carbs>,<fat>` line. `macros "chicken breast 100g"` -> `chicken breast
  100g,31,0,3`. This output is EXACTLY the `what,protein,carbs,fat` shape
  `journal_add_macros` consumes - the two tools chain.
- search: `macros -q "chick"` -> a text list ("Foods matching 'chick':\n\n  Chicken
  Breast g\n  ...") (exit 0).
- insert: `macros -i "banana 100g,1,23,0.3"` -> adds a food to the DB (a WRITE).
- unknown food: `Error: ... Unknown food: <food>\n\nTip: Use -q to search` (exit 1;
  `_run` folds stderr into the returned text, so the agent gets a usable message).
- NO `--json`; output is plain text. No config/env knob: the CLI resolves its DB at
  `$HOME/.local/share/nvim/macros.csv` itself (exists). So the scufris tool just
  shells out via `_run(["macros", ...])` - no SCUFRIS_* knob, unlike the den tools.
- Published with `overlays.default` (`pkgs.macros`), already applied in nix.dotfiles
  (flake/home-configurations.nix) - so a deployed-PATH follow-up mirrors the `today`
  one (separate task).

## Design decisions (this task)

- audience: ORCHESTRATOR-only, like the journal tools - it is the operator's personal
  food DB and pairs with `journal_add_macros`. No special gating needed: `apply_role`
  already strips every non-`request_input` tool for a sub-agent, so a project agent
  never sees these. (No env injection at all, so nothing to leak.)
- surface: one fine-grained tool per CLI mode (matches the one-tool-per-op pattern +
  strong descriptions), routed through `_run`. Names avoid clashing with the existing
  `journal_add_macros`.
- no config knob: `macros` self-resolves its DB; the tool needs no settings/env.

## Steps

- [x] Add a `macros_lookup(query)` tool: `_run(["macros", query])`; strong
      description (output feeds `journal_add_macros`; use for "macros for 2 eggs").
- [x] Add a `macros_search(query)` tool: `_run(["macros", "-q", query])` (fuzzy food
      search), for "what foods match X" / to find the exact name before a lookup.
- [x] Add a `macros_add_food(row)` tool: `_run(["macros", "-i", row])` (WRITE: add a
      food "name amount,protein,carbs,fat" to the DB), guarded on empty input.
- [x] Update `test_tools_registered` to include the new tools.
- [x] Add MCP tests driving the REAL `macros` CLI (skipif when absent, like the
      journal end-to-end tests) plus deterministic argv-capture tests (stub `_run`,
      always green incl. the nix flake check sandbox): lookup, search, add-food argv;
      empty-input guards.

## Definition of Done

1. The macros tools exist, are orchestrator-scoped, and each carries a description
   steering the model to use them (and noting lookup -> journal_add_macros).
   (cmd: `nix develop -c python -m pytest tests/test_mcp_server.py -q`)
2. argv-contract + guard tests pin the exact `macros`/`-q`/`-i` invocations
   deterministically (no `macros` binary needed); real-CLI tests run where `macros`
   is on PATH and skip in the sandbox.
   (cmd: `nix develop -c python -m pytest tests/test_mcp_server.py -q`)
3. Full local gate green: ruff, mypy, pytest.
   (cmd: `nix develop -c bash -c 'ruff check . && mypy . && python -m pytest -q'`)

## Notes

- Follow-up (separate task): put `pkgs.macros` on the DEPLOYED scufris service PATH in
  nix.dotfiles (mirrors the `today` deploy 20260726-225845), so these tools work off a
  dev box. File it, do not widen this task.
