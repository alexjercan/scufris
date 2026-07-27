# Retro: Split MCP into scufris + den + agent servers; per-server live health

- TASK: 20260727-105609
- BRANCH: feature/split-mcp-den
- REVIEW ROUNDS: 1 (APPROVE, out-of-context)

See TASK.md/NOTES.md for what changed and DECISION.md for the why; this is
process only.

## What went well

- Two upfront exploration passes (backend map + web/settings map) BEFORE planning
  meant the plan named exact files and line ranges, so the implementation had no
  "where does this live" dead ends across a 20-file diff.
- Confirmed all three load-bearing forks before building: split mechanism +
  health semantics via AskUserQuestion at understanding time, callback-server id
  at the plan gate. No mid-build shape surprise (the exact failure mode the flow
  skill warns about).
- Grepped every caller and test double of the renamed `scufris_mcp_server`
  (single -> list) up front (`protocol-signature-change-hits-the-doubles`), so the
  rename landed in one pass instead of a trail of `TypeError`s.
- The plan gate answer for health was "spawn each server"; realised in-process was
  the honest, consistent realisation and RECORDED that divergence in DECISION.md
  rather than silently diverging or blindly following the literal wording.
- Splitting the 1000-line `test_mcp_server.py` along the module split (den/agent/
  common test files) kept each test file to one module; moved tests verbatim and
  only re-pointed the `_run` monkeypatch target, so coverage is unchanged.
- Out-of-context round-1 review returned APPROVE with only NITs.

## What went wrong

- `nix flake check` failed TWICE on the same class of issue after all local fast
  checks were green. (1) ruff flagged I001 import-sort in a test file that
  `ruff format` had NOT fixed - `ruff format` does not sort imports; the linter
  does. (2) After fixing that in the working tree, the flake kept failing on the
  same stale file, because nix flakes only see git-TRACKED files - my new modules
  were untracked and invisible, so it was checking a tree without them. Root
  cause: I treated "local fast checks green" (which see the dirty working tree) as
  equivalent to the flake gate, forgetting both that `ruff format` != `ruff check
  --fix` and that the flake source is the git-tracked set, not the working dir.
  Cost: two wasted full `nix flake check` runs before the model clicked.
- `ruff check --fix .` (repo-wide) reflowed two UNRELATED files (agent_store.py,
  test_telegram.py) via formatter-version drift on master, widening the diff and
  forcing a revert. This is the THIRD occurrence of
  `format-only-the-files-you-edited-not-whole-dirs` - the same trap, now also via
  `ruff check --fix .` (not just `ruff format <dir>`).

## What to improve next time

- On a branch that ADDS files, `git add` (or commit) before running `nix flake
  check` - the flake ignores untracked files and will check a stale tree.
- Reach for `ruff check --fix <touched paths>` (catches I001 the writing formatter
  misses), scoped to the files you edited, never `.` or a whole dir - the repo-wide
  form reflows unrelated drift into the diff.

## Action items

- [x] Bumped `format-only-the-files-you-edited-not-whole-dirs` to x3 and moved it
  to Pending promotions (already tagged -> work skill verify-step).
- [x] Added ledger entries `nix-flake-check-sees-only-tracked-files` and
  `ruff-format-is-not-lint-fix`.
