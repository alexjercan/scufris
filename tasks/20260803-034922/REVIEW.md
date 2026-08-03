# Review: Pin the two legacy-diagnostics tests that cannot go red

- TASK: 20260803-034922
- BRANCH: fix/pin-legacy-diagnostics-tests

## Round 1

- REVIEWER: out-of-context
- VERDICT: APPROVE

- [ ] R1.1 (MINOR) tasks/20260803-034922/falsify.sh:51 - the harness never
  guarantees a clean tree. If `git apply -R` fails, or the run is interrupted
  between apply and revert, the sabotage stays applied to
  `scufris/api/legacy_agent.py` / `web/src/agent-view.ts`; the header's
  dirty-tree clause only promises a non-zero exit, not cleanup. Add an EXIT
  trap reverting `sabotage-*.patch`, plus a final
  `git diff --quiet -- scufris web || { echo "FAIL: tree left dirty"; exit 1; }`
  before the summary echo.
  - Response:
- [ ] R1.2 (MINOR) tasks/20260803-034922/sabotage-r21.patch:1 - the three
  short-circuits (usage, memory, account quota) ship as one patch, and pytest
  aborts at the first failing assertion (`tests/test_app.py:1882`, the usage
  one), so only the usage assertion is mechanically proven falsifiable while
  RETRO.md:36 reads as if all three are. Split into
  `sabotage-r21-usage.patch` / `-memory.patch` / `-account.patch` and add three
  `run_case` invocations, so each restored short-circuit is shown to turn the
  test red on its own.
  - Response:
- [ ] R1.3 (NIT) tasks/20260803-034922/falsify.sh:44 - the RED phase treats any
  non-zero exit as "red under sabotage", so a sabotage that broke import or
  collection would score identically to a real pin. Capture the run output and
  require it to contain the named test id with a failure marker (e.g.
  `grep -q "FAILED tests/test_app.py::test_disabled_agent_is_supported_not_unsupported"`)
  before counting the case as red.
  - Response:

Verification by the recording pass, re-run in the worktree:

- `bash tasks/20260803-034922/falsify.sh` -> exit 0, both cases red then green;
  `git status --porcelain` empty afterwards.
- `python -m pytest -q` -> exit 0. `cd web && npm run ci` -> exit 0.
  `tatr check` -> exit 0.
- Both `cmd:` proofs from `tatr proofs 20260803-034922` pass on their stated
  criterion. No `manual:` proofs, so no pending user checks.
- Re-derived independently, not taken from the reviewer: R1.2's premise. The
  RED transcript aborts at `tests/test_app.py:1882` with
  `TypeError: 'NoneType' object is not subscriptable`, so the `session_count`
  and `account["quota"]` assertions never execute under sabotage. Downgraded to
  MINOR rather than MAJOR because the diff itself shows each assertion
  discriminates - the old test asserted `session_count == 0` and
  `quota == {"supported": True, "value": None}`, the exact readings the
  short-circuit produces - so a split patch mechanizes a discrimination that is
  already evident, and any single-reader regression still turns this test red.
- Harness robustness probes that passed: a no-op sabotage is caught (test stays
  green -> FAIL); a stale vitest `-t` name exits 0, so the RED phase reports
  FAIL; a stale pytest node id exits 4 in both phases and is caught by the GREEN
  half. `-u -o pipefail` are set and expansions quoted; the absent `set -e` is
  deliberate, since failures are counted rather than aborted.
- Honesty: every Close-out and RETRO number re-ran and reproduced exactly. The
  `get_account` sabotage is a behavioural stand-in rather than a literal revert
  of `6da0c50`, and both TASK.md Notes and RETRO.md disclose that; the
  observable envelope is identical, which is what the test reads.
- Design: DECISION.md's claim that the deleted vitest case's coverage survives
  elsewhere holds - `web/src/chat-sidebar.test.ts:123,125` exercise
  `renderUsage(null)` and `primary: null`.
- Docs: no product code changed; the deleted test name appears only under
  `tasks/`, which is exempt. No stale README or AGENTS.md mentions.
- Not verified: `nix flake check` (no Nix-level change in the diff).
