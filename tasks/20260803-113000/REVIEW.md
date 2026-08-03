# Review: Prove the startup sweep clears a building row orphaned by a crash

- TASK: 20260803-113000
- BRANCH: test/orphaned-building-row-swept

## Round 1

- REVIEWER: out-of-context
- VERDICT: APPROVE

- [ ] R1.1 (MINOR) tests/test_nixos_config_change.py:699 - the new comment names
  `task 20260803-113000 and 20260803-014401 DECISION.md 1`, but `AGENTS.md:103`
  says task IDs belong in task records and Markdown, never in code comments, and
  the policy table (`AGENTS.md:99`) says to delete the lore and keep the
  invariant as a fact about the code. Drop the two ID references from lines
  699-700 and keep the load-bearing sentences: the generator writes `cancelled`
  before re-raising, so no clean shutdown can produce this row, and it must not
  be simplified back into a second HTTP build.

- Not a finding, context for R1.1: the immediate neighbour
  (`tests/test_nixos_config_change.py:632`) and the production docstring
  (`scufris/hostconfig/changes.py:158`) both carry the same ID references, so the
  diff follows local precedent rather than inventing it. Pre-existing lines are
  out of scope, and Step 3 requires the neighbouring test stay byte-identical.

- Process signal: Step 2 of the plan explicitly instructed naming both task IDs
  in the code comment, which contradicts `AGENTS.md`. The conflict entered at
  planning, not implementation; a plan step that mandates comment content should
  check the comment policy first.

Verified by the recording pass, not only by the reviewer:

- `ruff check . && ruff format --check . && mypy .` -> clean, 228 files.
- `python -m pytest` -> rc 0, one skip, no failures.
- `tatr check` -> rc 0.
- `abandon_builds` (`scufris/hostconfig/changes.py:146`) is the only writer of a
  BUILDING -> FAILED transition and the only source of the "the server restarted"
  error string (`grep -rn restart scufris/hostconfig/`), and the restarted app
  has no live run for the row, so the three assertions can be satisfied by the
  sweep and by nothing else.
- `AGENTS.md:85-104` re-read for R1.1; the policy line is literal.
- The diff touches one import line and adds one test function; the live-process
  test is byte-identical.

Pending user checks (`manual:`, not resolvable by review):

- Delete `config_changes.abandon_builds()` from `scufris/app.py:423`, run
  `python -m pytest tests/test_nixos_config_change.py -k orphaned`, confirm it
  fails, revert immediately.
- `git diff master -- tests/test_nixos_config_change.py` shows one changed import
  line and one added test function and nothing else.

Inspection commands:

```bash
cd "$(sprout show test/orphaned-building-row-swept)"
git diff master...HEAD
python -m pytest tests/test_nixos_config_change.py -k "orphaned or restart"
ruff check . && ruff format --check . && mypy . && python -m pytest
tatr check
```
