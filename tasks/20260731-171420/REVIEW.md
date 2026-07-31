# Review: Establish the file-size guard and sweep comment bloat

- TASK: 20260731-171420
- BRANCH: chore/file-size-guard

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) tasks/20260731-171420/TASK.md:150 - the recorded pytest
  total is a number no run produces: master is 881 and this branch is 895
  (881 + the 14 new guard tests), not the "883 passed" claimed. The same line
  class at TASK.md:153 records the drop-`app.py` falsification as `3769 lines,
  cap 600`; the guard now prints `3764`, because the sweep took 5 lines out of
  `app.py` after that falsification was run. Replace 883 with 895 and 3769
  with 3764.
  - Response: fixed in eee82a1 - 896 passed (881 on master + 15 guard tests, one added this round) and `3764 lines, cap 600`, both from a re-run rather than from memory.
- [x] R1.2 (MAJOR) tasks/20260731-171420/TASK.md:155 - `ruff format --check .`
  is recorded as clean; it exits 1 on 17 files, 5 of them files this diff
  edited (`scufris/app.py`, `checks.py`, `host_approvals.py`,
  `hostd/protocol.py`, `telegram.py`). It fails identically on master and is
  not part of the gate (`checks.ruff` runs `ruff check .` only), so it is
  pre-existing - but the evidence line asserts a result the command does not
  give. Either drop `ruff format --check .` from that line, or run
  `ruff format` on the five files this diff touched and keep the claim scoped
  to them.
  - Response: fixed in eee82a1 - the evidence line now records `ruff format --check .` as NOT clean (17 files, identically on master), notes the gate runs `ruff check .` only, and states the property this diff does control: no line it adds is over 88 columns.
- [x] R1.3 (MINOR) scripts/check_file_size.py:92 - `_is_skipped` applies the
  `result`/`result-` rule to every path component INCLUDING the filename, so a
  covered source file whose basename starts with `result-` is silently exempt
  from the cap. Verified: `_is_skipped("web/src/result-view.ts")` is `True`
  while `cap_for` gives it 600. Given this repo's `*-view.ts` naming that is a
  reachable hole. Test directory components only - `relative.split("/")[:-1]`.
  - Response: fixed in eee82a1 - `_is_skipped` now tests `relative.split("/")[:-1]`, and `test_covered_files_keeps_a_source_file_named_like_a_build_output` pins it with `web/src/result-view.ts` at cap+1.
- [x] R1.4 (MINOR) tasks/20260731-171420/TASK.md:158 - the DoD proof
  `nix flake check` does not pass on the branch. `checks.records` is green on
  master and red here purely because the record moved to `STATUS: IN_PROGRESS`
  under the 0.1.0 pin. The close-out describes this accurately and
  20260731-175511 owns the bump, so the fix is procedural: re-run
  `nix flake check` once the record reaches DONE and record THAT result,
  rather than closing on a red gate.
  - Response: fixed in eee82a1 - the evidence line now says the gate is re-run and its result re-recorded once this record reaches DONE, when the false positive no longer fires. 20260731-175511 still owns the pin bump.
- [x] R1.5 (MINOR) scufris/agent.py:155 - "the fix for R1.3 stripped codex's
  environment" and "(review round 2, R2.1)" now dangle: the citation that gave
  those finding ids their antecedent was deleted 11 lines above in this same
  diff. Review-finding ids are the same class of lore the sweep exists to
  remove. Rewrite as "the first fix stripped codex's environment and the claude
  backend went on spawning with no ``env=`` at all", with no finding ids.
  - Response: fixed in eee82a1 - the sentence now reads "the first fix stripped codex's environment and the claude backend went on spawning with no ``env=`` at all", with no finding ids.
- [x] R1.6 (MINOR) README.md:375 - the doc sweep missed the two places that
  enumerate the gate's checks, both stale now that `filesize` exists. Update
  `nix flake check   # ruff + mypy + pytest + task-record conformance` to name
  the file-size guard, and likewise `.github/workflows/ci.yaml:29`
  (`name: nix flake check (ruff / mypy / pytest / records)`) and its header
  comment at ci.yaml:5.
  - Response: fixed in eee82a1 - README.md:375 now reads `ruff + mypy + pytest + file-size guard + records`, and ci.yaml's header comment, job name and step comment all name the `filesize` check.
- [x] R1.7 (MINOR) tasks/20260731-171420/TASK.md:143 - the close-out says
  "Three real deferred items became `TODO:` one-liners"; the diff adds four
  (`scufris/config.py:164`, `scufris/backends.py:635`, `backends.py:808`,
  `backends.py:897`). Change three to four.
  - Response: fixed in eee82a1 - four, and each one is now named (opencode idle-timeout alignment, opencode SSE streaming, image attachments on claude and on opencode).
- [x] R1.8 (NIT) scufris/checks.py:242 - the reflow left a 116-character
  docstring line, over the repo's 88-column convention; `ruff format` does not
  rewrap prose, so nothing catches it. Rewrap the paragraph to 88.
  - Response: fixed in eee82a1 - rewrapped to 88.
- [x] R1.9 (NIT) scufris/agent.py:148 - several deletions left ragged
  half-width lines instead of a rewrapped paragraph; also
  `scufris/backends.py:352`, `scufris/projects.py:5`, and the `write` /
  `opt-in.` break in `scufris/agent_store.py`. Rewrap each edited paragraph.
  - Response: fixed in eee82a1 - rewrapped the paragraphs in agent.py, backends.py, projects.py and agent_store.py. Verified across the whole diff: no line this branch adds to scufris/, web/src/ or scripts/ exceeds 88 columns.
- [x] R1.10 (NIT) scripts/check_file_size.py:99 - `covered_files` rglobs the
  whole repo (19,403 paths, 0.59s) to keep 156, walking `.git` and
  `web/node_modules` in full. Iterate the three `COVERED_ROOTS` instead
  (`for name in COVERED_ROOTS: (root / name).rglob("*")`), which also removes
  the need for the repo-root-only `result*` rule in R1.3.
  - Response: fixed in eee82a1 - `covered_files` now iterates `COVERED_ROOTS` and rglobs each, so `.git` and `web/node_modules` are never walked.

### Verified this round

- Guard: `python scripts/check_file_size.py` exit 0, and falsified in both
  directions against the real tree (dropping `scufris/app.py` reports
  `3764 lines, cap 600`; adding `web/src/markdown.ts` reports a stale entry).
- `nix build .#checks.x86_64-linux.filesize` builds. `checks.records` fails on
  the branch and passes on master (R1.4).
- `python -m pytest` 895 passed (881 on master, +14). `ruff check .` clean,
  `mypy .` clean, `cd web && npm run ci` green. `ruff format --check .` exits
  1, on master too (R1.2).
- Grep proofs: `rg "2026[0-9]{4}-[0-9]{6}" scufris web/src -g '!*.md'` and
  `rg "XXX|HACK" scufris web/src` both empty; `rg "600" AGENTS.md` present.
- Allowlist is 23 path-only entries matching the Notes (10 py + 4 ts source,
  7 py + 2 ts test); every entry resolves to a non-None cap.
- Steps 1-10 re-read against the diff: all delivered literally, including the
  `web/src/project-detail-view.test.ts:41` fixture rename.
- New tests assert behavior: each of the three ratchet rules has a test that
  inverts if its rule is deleted, and the live-tree test is the one CI fires.
- Re-derived in-session: R1.2 (`ruff format --check .` exit 1, 17 files),
  R1.3 (`_is_skipped("web/src/result-view.ts") is True` with cap 600), R1.5
  (the dangling ids at `scufris/agent.py:155-157`), and R1.6 (the stale check
  enumerations in `README.md` and `.github/workflows/ci.yaml`).

### Pending user checks

- `manual:` inspect `rg -n "TODO|FIXME|BUG|NOTE|XXX|HACK" scufris web/src` -
  9 hits, all one of the four markers, 4 of them added here. `XXX` and `HACK`
  remain absent. Not resolvable by the review side.

## Round 2

- REVIEWER: out-of-context
- VERDICT: APPROVE

All ten round-1 findings verified fixed and ticked above. Two nits opened on
the fixes themselves; neither blocks the verdict.

- [ ] R2.1 (NIT) tasks/20260731-171420/TASK.md:155 - the replacement evidence
  line asserts "no line this diff adds is over 88 columns" without scope, but
  the diff adds three `AGENTS.md` comment-policy table rows at 89, 93 and 123
  columns (a Markdown table row cannot be wrapped). Scope the claim to the code
  it controls: "no line this diff adds to `scufris/`, `web/src/`, `scripts/` or
  `tests/` is over 88 columns".
  - Response:
- [ ] R2.2 (NIT) .github/workflows/ci.yaml:5 - the R1.6 edit inserted its new
  clause without rewrapping, leaving "# repository conformance, and" as a
  ragged half-width line - the same defect R1.9 flagged in the Python
  docstrings. Rewrap the four-line header paragraph.
  - Response:

### Verified this round

- Each of R1.1-R1.10 checked against the code, the record and re-run command
  output rather than against its Response line. All ten confirmed.
- `python -m pytest` 896 passed; `--collect-only` gives 896 on the branch and
  881 on master, the delta being the 15 cases in `tests/test_check_file_size.py`.
- The corrected Evidence figures match the commands digit for digit: 896, and
  `scufris/app.py: 3764 lines, cap 600` from the drop-`app.py` falsification.
- `ruff check .` clean, `mypy .` clean (117 files), `cd web && npm run ci`
  green (prettier, eslint, 258 vitest cases, webpack). `ruff format --check .`
  exits 1 with 17 files on the branch AND on master, as the record now states.
- `python scripts/check_file_size.py` exit 0;
  `nix build .#checks.x86_64-linux.filesize` builds; `nix flake check` exits 1
  with `checks.records` as the sole failure, on the stale-pin message.
- R1.10 checked for behavior equivalence, not just speed: old and new
  `covered_files` return the identical 156-path set against the real tree
  (empty symmetric difference), at 0.004s versus 0.499s.
- Re-derived in-session: the R1.3 fix inverts as claimed (with `_is_skipped`
  restored to matching every component, `covered_files` on a tree holding only
  `web/src/result-view.ts` returns `[]` instead of that path, so the new test
  fails); and both R2.1 and R2.2 confirmed by direct inspection.

### Pending user checks

- `manual:` inspect `rg -n "TODO|FIXME|BUG|NOTE|XXX|HACK" scufris web/src` -
  9 hits, all one of the four markers, 4 of them added here. `XXX` and `HACK`
  remain absent. Not resolvable by the review side.
- R1.4's promise - re-run `nix flake check` and re-record its result once this
  record reaches DONE - is future work and not observable now.
