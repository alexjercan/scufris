# Retro: macros food-lookup MCP tools

- TASK: 20260727-010447
- BRANCH: feature/macros-mcp-tools
- REVIEW ROUNDS: 1 (out-of-context; APPROVE, 2 NITs left as-is)

See TASK.md for the CLI contract; process only here.

## What went well

- Reused the pattern established two tasks ago (the journal tools): one tool per CLI
  mode through `_run`, orchestrator-only via role scoping (no new gating), and the
  two-layer tests (deterministic argv/guard + `skipif(shutil.which('macros'))`
  real-CLI). The whole feature was a fast, low-risk copy of a known-good shape.
- Made the real-CLI tests HERMETIC instead of leaning on the operator's live DB:
  the `macros` CLI resolves its csv from `$HOME`, and `_run` inherits the env, so a
  temp-HOME fixture with a seeded macros.csv redirects it. That removed a fragile
  dependency on "egg exists in the user's DB" AND let `add_food` (a write) be tested
  safely against the temp copy - the reviewer confirmed the operator's real
  macros.csv was untouched.
- No config knob needed: checked how `macros` resolves its DB before assuming a
  den_path-style setting, and found it self-resolves - so the tool stayed a plain
  `_run` wrapper. Understanding-first avoided inventing a needless SCUFRIS_* knob.

## What went wrong

- Nothing of substance. Two NITs (a bare-flag query is parsed as a flag; an insert
  echo assertion), both degenerate and left as-is with rationale.

## What to improve next time

- Keep reaching for the hermetic-temp-env trick when wrapping a CLI whose data path
  is env-derived: seed a temp store and redirect via the CLI's own env var, rather
  than reading/writing the operator's live data in tests.

## Action items

- [x] Delivered: macros_lookup / macros_search / macros_add_food, orchestrator-only,
  with argv + real-CLI (hermetic) tests.
- [x] Ledger: added `wrap-env-derived-cli-with-a-temp-home-fixture`.
- [x] tatr 20260727-011526 (follow-up): put `pkgs.macros` on the deployed scufris
  service PATH in nix.dotfiles, mirroring the `today` deploy.
