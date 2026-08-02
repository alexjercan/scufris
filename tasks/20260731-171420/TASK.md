# Establish the file-size guard and sweep comment bloat

- PRIORITY: 95
- TAGS: chore, v0.2.0, maintainability, kiss
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260731-171411

## Story

As a maintainer, I want an enforced file-size cap and one comment policy
applied across the codebase, so that context cost stops growing silently and
later splits have a gate that fails when they regress.

## Steps

- [x] Add `scripts/check_file_size.py` (stdlib only; must pass `ruff check .`
      and `mypy .`, which both cover `scripts/`). Coverage and caps:
      `scufris/**/*.py` and `web/src/**/*.ts` (excluding `*.test.ts`) at 600
      lines; `tests/**/*.py` and `web/src/**/*.test.ts` at 900. Skip
      `__pycache__`, `node_modules`, `.venv`, `result*`. `.css`, `.html`, and
      `.json` are not covered (see DECISION.md).
- [x] Make the allowlist a path-only `frozenset` of repo-relative paths. Fail
      with each offender's path, line count, and cap. Also fail on a stale
      entry: an allowlisted file now inside its cap must be removed from the
      list. That is the ratchet - entries may only leave.
- [x] Add `tests/test_check_file_size.py`, following `tests/test_release.py` as
      the precedent for testing a `scripts/` module: oversized file flagged,
      allowlisted oversized file accepted, stale entry rejected, and the real
      tree passing.
- [x] Seed the allowlist with every current offender (15 source, 9 test; see
      Notes). Every entry is owned by a sibling child task, which removes it.
- [x] Wire the guard into `flake.nix` `checks` next to `records`:
      `filesize = mkCheck "filesize" "python scripts/check_file_size.py";`
- [x] Sweep `scufris/` for comments and docstrings citing a task, spike, or
      decision ID (95 lines across ~50 files). Delete the lore; keep the
      invariant it wrapped, restated as a fact about the code. Do not delete a
      docstring to satisfy the cap.
- [x] Sweep `web/src/*.ts` the same way, and rename the fixture literal
      `id: "20260720-120000"` in `web/src/project-detail-view.test.ts:41` to a
      non-ID-shaped string so the grep proof can be clean.
- [x] Compact real deferred work into `TODO:`/`FIXME:`/`BUG:`/`NOTE:`
      one-liners. Delete comments that restate the code.
- [x] Re-run the guard and prune allowlist entries for files the sweep dropped
      under their cap.
- [x] Record the 600/900 cap and the epic comment policy table in `AGENTS.md`.

## Definition of Done

- The guard flags an oversized file
  (test: `test_check_file_size_flags_oversized_file`).
- The guard flags a stale allowlist entry
  (test: `test_check_file_size_rejects_stale_allowlist_entry`).
- The guard passes on the tree (cmd: `python scripts/check_file_size.py`).
- The guard runs in the canonical backend gate (cmd: `nix flake check`).
- No comment or docstring cites a task/spike/decision ID
  (cmd: `rg -n "2026[0-9]{4}-[0-9]{6}" scufris web/src -g '!*.md'`).
- Behavior unchanged (cmd: `python -m pytest`, cmd: `cd web && npm run ci`).
- `AGENTS.md` states the cap and the comment policy
  (cmd: `rg -n "600" AGENTS.md`).
- Every retained deferred-work comment uses one of the four markers
  (manual: inspect `rg -n "TODO|FIXME|BUG|NOTE|XXX|HACK" scufris web/src`;
  `XXX` and `HACK` are already absent, so this asserts the sweep added none).

## Notes

- Epic: 20260731-171411. Do not combine with any split; this task changes
  comments and adds a gate.
- Source offenders to seed (lines): `scufris/app.py` 3769,
  `scufris/telegram.py` 1448, `scufris/backends.py` 1098,
  `scufris/agent_store.py` 1035, `scufris/sessions.py` 835,
  `scufris/agent.py` 832, `scufris/hostd/actions.py` 774,
  `scufris/hostconfig.py` 664, `scufris/mcp_host_tools.py` 630,
  `scufris/auth.py` 608, `web/src/agent-chat-view.ts` 1106,
  `web/src/host-view.ts` 1022, `web/src/stats-view.ts` 870,
  `web/src/common.ts` 834.
- Test offenders to seed: `tests/test_app.py` 3813, `tests/test_telegram.py`
  1760, `tests/test_host_action_api.py` 1285, `tests/test_auth.py` 1219,
  `tests/test_host_inspection.py` 1076, `tests/test_nixos_config_change.py`
  1044, `tests/test_agent_store.py` 937,
  `web/src/agent-chat-view.test.ts` 1183, `web/src/host-view.test.ts` 997.
- Every seeded entry has an owning sibling: 171428 (agent runtime), 171429
  (telegram), 171430 (host/hostd/auth/hostconfig/mcp_host_tools), 171431
  (frontend views), 171432 (test suites), 20260729-103712 (`app.py`,
  `test_app.py`).
- `web/src/style.css` (2662) is the only offender with no owning split task,
  which is why the guard does not cover `.css` (DECISION.md).
- `scufris/hostd/README.md` links task DECISION.md files on purpose; the grep
  proof excludes Markdown. Epic DoD 4 excludes only `*.json` and should be
  read as `-g '!*.md'` too.
- `tests/**` carries ~23 ID citations. Out of scope here (epic DoD 4 greps
  `scufris web/src`); 171432 applies the policy to the files it touches.
- Assumption: `mkCheck` provides the dev venv's `python`, so the guard needs no
  new Nix input.

## Close-out

### What and why

`scripts/check_file_size.py` walks `scufris/**/*.py`, `tests/**/*.py` and
`web/src/**/*.ts`, caps source at 600 lines and tests at 900, and runs as the
`filesize` check in `nix flake check`. Its `ALLOWLIST` is a `frozenset` of 23
repo-relative paths and it fails in BOTH directions: over the cap unlisted, and
listed while inside the cap (or naming a file that is not there). The second
rule is the ratchet - without it a split can land while leaving its entry
behind, and the file is free to grow again.

The sweep removed all 93 task/spike/decision ID citations from `scufris/` and
`web/src/`, keeping the invariant each one wrapped as a statement about the
code. Four real deferred items became `TODO:` one-liners (opencode idle-timeout
alignment, live token streaming over the opencode SSE bus, and image
attachments on each of the claude and opencode backends). The
`AGENTS.md` "File size and comments" section states the caps, the ratchet rule,
and the epic's comment policy table.

### Alternatives

- Per-file line budgets in the allowlist. Rejected in DECISION.md: churn on
  every edit, and it still lets a file sit one line under its recorded budget.
- Covering `.css`. Rejected in DECISION.md: `web/src/style.css` (2662) has no
  owning split task, so the entry would be permanent and the allowlist would
  become a config knob.
- A `--root` flag on the guard so tests could drive it against a fixture tree.
  Rejected: the module allowlist is about THIS repo, so a foreign root would
  read every entry as stale. The tests monkeypatch `REPO_ROOT`/`ALLOWLIST`
  instead, and `check(root, allowlist)` takes both explicitly.

### Difficulties and diagnosis

Two things the plan did not predict.

1. `nix flake check` first failed with `can't open file
   '/build/work/scripts/check_file_size.py'`. The flake's `src = ./.` is
   git-filtered, so a new UNTRACKED file is invisible to every check. `git add`
   before running the gate.
2. The `records` check then failed with `20260731-171420:
   unplanned-in-progress: IN_PROGRESS task lacks PLAN STATUS: APPROVED` against
   a record that plainly carries that field. `flake.nix` pins tatr 0.1.0, which
   predates the v2 schema commit 9d78ebe migrated the records to, so it cannot
   parse `PLAN STATUS` at all. Bumping the input to 0.2.0 cleared every
   task-record finding but surfaced 11 LESSONS.md ledger findings, 10 of them
   promotion decisions AGENTS.md reserves for the operator. Per the operator's
   call the pin stays as it is here and 20260731-175511 owns the bump plus the
   dispositions.

### Evidence

All figures below are from the round-2 re-run, after the review-round-1 fixes.

- `python scripts/check_file_size.py` - exit 0 on the tree.
- `python -m pytest` - 896 passed, including 15 in
  `tests/test_check_file_size.py` (881 on master).
- Guard falsified in both directions against the real tree: dropping
  `scufris/app.py` from the allowlist reports `3764 lines, cap 600`; adding
  `web/src/markdown.ts` reports it as a stale entry.
- `ruff check .` and `mypy .` - clean. `ruff format --check .` is NOT clean
  (17 files), identically on master; it is not part of the gate, which runs
  `ruff check .` only, and no line this diff adds to `scufris/`, `web/src/`,
  `scripts/` or `tests/` is over 88 columns (R2.1: three `AGENTS.md` table rows
  do exceed it, and a Markdown table row cannot be wrapped).
- `cd web && npm run ci` - prettier, eslint, 258 vitest cases and the webpack
  build pass.
- `rg -n "2026[0-9]{4}-[0-9]{6}" scufris web/src -g '!*.md'` - no matches.
- `rg -n "XXX|HACK" scufris web/src` - no matches.
- `rg -n "600" AGENTS.md` - the cap is recorded.
- `nix flake check` - **all checks passed**, re-run after this record reached
  DONE, as R1.4 required. While the record was IN_PROGRESS the `records` check
  failed on the stale-pin false positive described above (the other four built
  throughout); a CLOSED record parses under tatr 0.1.0, so the whole gate -
  `filesize`, `ruff`, `mypy`, `pytest`, `records` - is green. The pin is still
  wrong for the duration of any task; 20260731-175511 owns it.

### Reflection

- The Steps said 15 source offenders; the tree has 14 (10 Python, 4
  TypeScript), so the allowlist holds 23 entries rather than 24. Counted from
  `wc -l`, not from the plan.
- No file dropped under its cap from the sweep alone (`scufris/auth.py` went
  608 -> 606), so nothing was pruned. Comment bloat was never the reason these
  files are oversized, which is the case for the split children.
- Next time, `git add` new files before the first `nix flake check`. A
  git-filtered flake source turns "the check does not exist" into "the file
  does not exist", and the error names the sandbox path rather than the cause.
