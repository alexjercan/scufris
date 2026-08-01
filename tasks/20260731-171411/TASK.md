# EPIC: Fit every component in one context

- STATUS: OPEN
- PRIORITY: 125
- TAGS: goal, epic, v0.2.0, refactor, maintainability, kiss
- KIND: EPIC
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT

## Epic

Make every Scufris source file readable inside a single implementation context
and strip comment bloat that costs context without changing behavior. The
codebase has grown to 24.7k Python lines and 7.5k frontend lines, with ten
source files and nine test files above the context-friendly cap, plus roughly
forty comments whose payload is task-record lore rather than code meaning.

Scope is KISS only: split oversized files along real seams, delete or compact
comments, and keep public docstrings. No behavior change, no new features, no
new abstraction that a single caller does not demand.

Cap: source files at 600 lines, test files at 900 lines. A 600-line Python file
is roughly 8-9k tokens, so a task can hold four to six in-scope files plus the
full flow skill inside a 150k implementation budget.

Comment policy for this epic:

| Comment | Action |
|---------|--------|
| Module/class/function docstring stating purpose and contract | Keep, trim to the contract |
| Guards a value, explains a non-obvious setting or invariant | Keep verbatim |
| Names a task/spike/decision ID as the only justification | Delete the lore, keep the invariant if one exists |
| Narrates what the code already says | Delete |
| Real deferred work or defect | Compact to `TODO:`/`FIXME:`/`BUG:`/`NOTE:` one-liner |

## Done Means

1. No non-generated source file under `scufris/` or `web/src/` exceeds 600
   lines and no file under `tests/` or `web/src/**.test.ts` exceeds 900 lines,
   except entries in an explicit shrinking allowlist
   (cmd: `python scripts/check_file_size.py`).
2. The size guard runs in the canonical backend gate and fails a regression
   (cmd: `nix flake check`).
3. The allowlist is empty except `scufris/app.py` and `tests/test_app.py`,
   which task 20260729-103712 owns
   (cmd: `rg -n "app" scripts/check_file_size.py`).
4. No source comment cites a task, spike, or decision ID as its justification
   (cmd: `rg -n "2026[0-9]{4}-[0-9]{6}" scufris web/src --glob '!*.json'`).
5. Every deferred-work comment uses one of `TODO:`, `FIXME:`, `BUG:`, `NOTE:`
   (cmd: `rg -n "TODO|FIXME|BUG|NOTE|XXX|HACK" scufris web/src`).
6. Behavior is unchanged: both canonical gates and both package builds pass
   (cmd: `nix flake check && cd web && npm run ci`).
7. `AGENTS.md` records the size cap and the comment policy so future tasks
   inherit them
   (cmd: `rg -n "600|comment policy" AGENTS.md`).
8. manual: after the epic, opening any single component for a change requires
   reading files that fit comfortably in one implementation context.

## Child Tasks

- [x] 20260731-171420 (p95) establish the size guard, comment policy, and
      repo-wide comment sweep
- [x] 20260731-171428 (p90) split the agent runtime modules
- [x] 20260731-171429 (p85) split the Telegram surface
- [x] 20260731-171430 (p80) split the host, hostd, and auth modules
- [x] 20260731-171431 (p75) split the oversized frontend views
- [x] 20260731-171432 (p70) split the oversized test suites

## Decisions

- Cap and comment policy decided in this record; children do not relitigate.
- `scufris/app.py` and `tests/test_app.py` stay with 20260729-103712 under epic
  20260729-102145. This epic only ratchets them in the allowlist.
- `scufris/auth.py` is SPLIT, not trimmed (20260731-171430, with the
  measurement): the sweep was worth 4 lines against a 6-line gap, and clearing
  the cap by one line is not clearing a ratchet. The deny-by-default middleware
  was never in that file - it is one middleware in `scufris/app.py` - and
  `auth/policy.py` is now the single module answering every question it asks.

## Fog

- Whether `web/src/common.ts` splits by concern or becomes a directory.

## Out of Scope

- `scufris/app.py` and `tests/test_app.py` (owned by 20260729-103712).
- Behavior changes, new features, dependency changes, and API surface changes.
- Reformatting or renaming beyond what a split requires.
- Introducing a docs site; docstrings are kept publishable, not published.

## Manual Acceptance

- (pending) after all children land: a component change reads naturally without
  opening a 1000-line file.
- (pending, from 20260731-171420) every retained deferred-work comment uses one
  of the four markers: inspect `rg -n "TODO|FIXME|BUG|NOTE|XXX|HACK" scufris
  web/src`. `XXX` and `HACK` are absent, so this asserts the sweeps added none.
