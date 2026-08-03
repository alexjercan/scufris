# Review: Clear the round-1 MINOR findings from the diagnostics alignment

- TASK: 20260803-042958
- BRANCH: chore/diagnostics-minors

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) CHANGELOG.md:26 - the new three-state bullet, the one
  deliverable of Step R1.3, is itself stale in the way the Story exists to
  remove. It says the wording is shared by the Telegram `/settings` summary,
  `health`, `usage` and `memory` cards. Telegram has no `memory` card
  (`SETTINGS_USAGE` is `/settings [health|usage|tools]`, `text.py:85`), and
  `render_health` (`render.py:286-307`) never emits `CAP_UNSUPPORTED` or
  `CAP_EMPTY` - `_quota_reading` has exactly two callers, `render_usage:329`
  and `render_settings_summary:386`. It also contradicts the surface table at
  `scufris/README.md:436-439`, which this task treats as the source of the
  vocabulary. Replace the surface clause with "shared by the Telegram
  `/settings` summary and `/settings usage` and by the web agent settings
  page's usage and memory panels".
  - Response: fixed in the round-2 commit. Confirmed both claims before editing:
    `_quota_reading` has exactly the two callers the finding names
    (`render.py:329`, `render.py:386`), `render_health` (`render.py:286-307`)
    emits no `CAP_*` string, and `rg -n 'memory' scufris/telegram/render.py`
    finds no memory card - only `_fmt_bytes`'s docstring and a host-stats
    line. Took the proposed replacement clause verbatim; the bullet now agrees
    with `scufris/README.md:436-439`.

- [x] R1.2 (MINOR) scufris/telegram/render.py:332 - the R1.4 rewrite leaves an
  89-char line, one over the repo's `line-length = 88` (`pyproject.toml:106`).
  `ruff format --check .` reports `1 file would be reformatted` on this branch
  and `229 files already formatted` on master, so this diff introduces the
  repository's only formatter violation. `AGENTS.md:53` names `ruff format .`
  as a required command; `nix flake check` runs `ruff check .` only
  (`flake.nix:233`), so the gate did not catch it. Run `ruff format` on the
  file - it splits the tuple across lines and changes nothing else.
  - Response: fixed in the round-2 commit. Ran `ruff format scufris/telegram/render.py`;
    it split the window tuple across lines and touched nothing else.
    `ruff format --check .` now reports `229 files already formatted`.

- [x] R1.3 (NIT) web/src/agent-settings-view.ts:62 - the R1.1c rewrap fixed the
  grep but left a 95-char line in a comment block whose other lines wrap near
  80 (line 60 is 81, line 61 is 67). Rewrap as ``// `capabilityText` in
  `agent-settings-panels.ts`). Null is neither -`` / `// it is a failed fetch.`
  so the block stays uniform and the token pair stays on one line.
  - Response: fixed in the round-2 commit. Split after `Null is neither -` rather than
    before it, which serves the same two constraints: the block is now 81/67/
    73/28 chars (was 81/67/95) and `capabilityText` still shares a line with
    `agent-settings-panels.ts`, so DoD proof 2c stays empty.

Verified in the worktree, both by the out-of-context reviewer and re-derived
here:

- All nine Steps re-read against the literal diff. Every ticked step is
  delivered, including the R1.6 carve-out: `statusPanel` keeps its `export` and
  is still imported at `agent-settings-view.ts:32,215`.
- Proofs 2-6 (the six greps) run verbatim; each returns exactly what the DoD
  names. `function resetsIn` is one hit, `web/src/common.ts:179`.
- Proof 1 falsified independently in a scratch copy: reverting only
  `usage.primary or usage.secondary` back to `usage.primary` makes the new case
  fail printing `usage: nothing reported yet`, the exact string the record
  claims. The test asserts behavior - percent, the `daily` label, and
  `CAP_EMPTY not in` - not mere execution.
- Proof 7 re-run here: `python -m pytest` 1108 passed / 1 skipped;
  `ruff check .` and `mypy .` clean (229 source files); `cd web && npm run ci`
  format:check, lint, tests and the webpack build all green. The close-out's
  numbers match.
- `render_usage`'s new early guard is exactly the old `not windows` condition;
  `plan_type` and the primary -> secondary ordering are unchanged, so the
  refactor is behavior-preserving and R1.2's case still agrees with it.
- No existing test weakened or deleted. No web test imported the two symbols
  that lost `export`, so the change costs no coverage.
- Doc sweep on `capabilityText`, `resetsIn` and `agent-settings-panels` across
  `README.md`, `scufris/README.md`, `web/README.md` and `AGENTS.md`: no stale
  mentions outside R1.1.
- No `manual:` proofs on this task, so there are no pending user checks.

Not verified: rendered Telegram and web output against a live backend. All
surface claims were checked by reading the renderers plus the pytest and jsdom
suites.

## Round 2

- REVIEWER: out-of-context
- VERDICT: APPROVE

- [ ] R2.1 (NIT) tasks/20260803-042958/TASK.md:204 - the round-2 Alternatives
  paragraph says R1.3 "proposed splitting before `Null is neither`; split after
  it instead". R1.3 proposed ``// `capabilityText` in
  `agent-settings-panels.ts`). Null is neither -`` / `// it is a failed fetch.`
  - the split AFTER it, which is character-identical to what shipped
  (`agent-settings-view.ts:62-63`). The record describes a divergence that did
  not happen. Replace the paragraph with "None - took R1.3's proposed wrap
  verbatim", or drop the R1.3 entry from Alternatives.
  - Response: correct - the delivered wrap is character-identical to the
    proposal. Replaced the paragraph with "Alternatives. None - R1.3's proposed
    wrap was taken verbatim", keeping one clause noting the correction. Prose
    only; no code touched, so the check suite is unaffected. Box left unticked:
    the round's reviewer is out-of-context and the in-session pass does not
    self-confirm.

All three round-1 findings confirmed fixed by the out-of-context reviewer and
re-derived here:

- R1.1: the CHANGELOG bullet now carries the proposed clause verbatim and
  agrees with `scufris/README.md:436-439`. `_quota_reading`'s two callers
  (`render.py:329,389`) and `render_health`'s `CAP_*`-free body re-checked.
- R1.2: `ruff format` split the window tuple at `render.py:331-334` and touched
  nothing else. `ruff format --check .` is clean.
- R1.3: fix delivered. Re-derived independently here - `awk length` on
  `agent-settings-view.ts:59-64` gives 81/67/73/28, and `capabilityText` still
  shares line 62 with `agent-settings-panels.ts`, so DoD proof 2c stays empty.
  The wrap is the finding's own proposal, hence R2.1.
- The round-2 commit touches only CHANGELOG prose, one formatter reflow and one
  comment rewrap. `render_usage`'s guard, `render_settings_summary`'s
  `usage.primary or usage.secondary`, the `export` drops and the `resetsIn`
  move are byte-unchanged since round 1. No regression.
- Checks re-run by the primary: `python -m pytest` 1108 passed, 1 skipped;
  `ruff format --check .` 229 files already formatted; `ruff check .` all
  checks passed; `mypy .` success, 229 source files; `cd web && npm run ci`
  exit 0 with the webpack build green. The reviewer additionally ran
  `nix flake check` (all checks passed) and all seven DoD proofs, including
  falsifying proof 1 in a scratch copy.
- No `manual:` proofs on this task, so there are no pending user checks.

Not verified: rendered Telegram and web output against a live backend.
