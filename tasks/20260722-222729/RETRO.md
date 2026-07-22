# Retro: T3 - prune the MCP surface (drop tatr_* tools)

- TASK: 20260722-222729
- BRANCH: refactor/prune-mcp-tatr-tools
- REVIEW ROUNDS: 2 (R1 REQUEST_CHANGES with 1 MINOR; R2 APPROVE in-session after fix)

See TASK.md for what changed and why; this is process only.

## What went well

- The removal sweep reached past the plan's named Steps: config.py comments,
  test_app.py's tools-endpoint tests, and a `web/` render fixture all referenced
  the tools. Grepping the worktree up front caught them in one pass - the frontend
  fixture especially, which the Python suite would never have flagged.
- Deleted the seven tatr tests wholesale (the tools are gone) rather than weakening
  them; kept `test_tools_registered` an exact `==` set so the assertion still pins
  the real surviving tools. The reviewer confirmed this.
- Distinguished the MCP tool `tatr_new` (removed) from `test_projects._tatr_new`
  (a local tatr-CLI fixture helper, legitimately kept) instead of blindly deleting
  every match.

## What went wrong

- R1.1: `.env.example` still had `["tatr_new"]`. Root cause: my up-front sweep used
  `grep -rn ... --include=*.py --include=*.md --include=*.ts ...`, and the
  extension globs silently excluded the extensionless `.env.example`. The stale
  sample survived and the out-of-context review caught it. My self-reflection had
  also overstated that the one-pass grep caught everything - corrected in TASK.md.

## What to improve next time

- An absence-proving sweep must be scoped by PATH (`--exclude-dir`), never narrowed
  by `--include=*.ext` globs - config often lives in dotfiles/extensionless files
  (`.env.example`, `Dockerfile`, `Makefile`). New ledger lesson added.

## Action items

- [x] Added `absence-grep-must-not-be-extension-scoped` (x1) to LESSONS.md.
- No follow-up code tasks. SPIKE Q4 permission-mode caveat remains an open question
  for the (deferred) Telegram bot work, not a code change here.
