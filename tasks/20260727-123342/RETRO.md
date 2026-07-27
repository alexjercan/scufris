# Retro: Remove MCP servers config and Profiles from settings

## What was delivered

Removed the "MCP servers" operator-config card and the "Profiles" named-config
feature from the orchestrator settings page, UI and backend, in one squash
commit. Kept the separate "MCP tools" health/catalog section and the built-in
scufris/den/agent servers. Collapsed the profile-shaped settings store to a flat
overrides file with a load-time migration. 542 backend + 186 web tests green;
ruff/mypy/webpack clean. Reviewed out-of-context: APPROVE, 0 findings.

## What went well

- The one ambiguity that mattered (two MCP cards: "MCP servers" vs "MCP tools")
  was surfaced to the user BEFORE building via AskUserQuestion, with a preview
  showing exactly which card each option removed. The user picked "just MCP
  servers" - had I inferred, I might have ripped out the wrong (or both) cards.
  The second question (UI-only vs full backend) turned a small UI edit into the
  correct full-stack removal. Confirming the artifact paid off.
- Two parallel Explore agents (frontend map, backend map) plus my own reads gave
  a complete, line-accurate removal map before any edit. The critical finding -
  that the kept "MCP tools" path is fed by `mcp_health.servers_for_audience`
  (built-in servers), NOT the operator `settings.mcp_servers` list - de-risked
  the whole change: removing the operator config could not break the kept view.
- The profile-shaped persistence format was the real trap. Writing a migration
  (`_overrides_from_persisted`) plus two migration tests meant the user's
  existing `settings.json` keeps its active settings instead of silently
  resetting.

## What went wrong / difficulties

- `ruff format scufris tests` reformatted two files I never touched
  (`agent_store.py`, `test_telegram.py`) - pre-existing line-length drift. Caught
  it at `git add` and reverted those two so the commit stays focused. Running the
  formatter over the whole tree, not just changed files, pulls in unrelated
  churn.
- `ruff check` caught an unused `import re` in BOTH `agent.py` and `app.py` after
  removing the only `re.fullmatch`/`re.compile` call sites. The grep I used to
  pre-check (`\bre\.[a-z]`) missed the `app.py` one because I only ran it against
  `agent.py`. Lint is the backstop, but grepping every file that lost its last
  regex use would have saved a round.
- The similarly-named `scufris_mcp_servers` (built-in core, KEEP) vs
  `settings.mcp_servers` (operator config, REMOVE) is an easy place to cut the
  wrong thing. Pinning the distinction explicitly in the plan and grepping with
  `grep -v scufris_mcp_servers` kept them straight.

## Lessons for future sessions

- When a repo-wide formatter is part of the check suite, run it against the
  changed paths only (or `git stash` unrelated drift), then `git diff --cached`
  before committing - unrelated format-only churn should not ride in a feature
  commit. (Reinforces the existing format-only lesson in LESSONS.md.)
- After deleting the last user of a stdlib import (here `re`), grep EVERY file
  that used it, not just the obvious one - `ruff check` will catch it but it
  costs a round.
- For a removal that straddles two similarly-named concepts, write the KEEP/
  REMOVE split into the plan verbatim and use `grep -v <keep-symbol>` in the
  done-check, so the audit does not re-flag the kept machinery.
- `web/node_modules` is a symlink into the main checkout; `git add -A` stages it.
  Explicitly `git reset HEAD web/node_modules` before committing in a sprout
  worktree. (Reinforces the node_modules-in-sprout lessons.)
