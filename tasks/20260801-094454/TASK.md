# Repoint the stale test-file citations left by the suite split

- PRIORITY: 0
- TAGS: refactor,backlog,maintainability,docs
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Story

As a maintainer, I want the two comments that still name pre-split test files
to name the files that actually hold those assertions, so that following a
citation lands on the pin rather than on an empty file.

Both are open, non-blocking findings from 20260731-171432's review round 1
(R1.2 and R1.3). Comment text only - no test moves, no behavior change.

## Steps

- [ ] `tests/conftest.py:181` - the `_Helper` section comment names
      `test_host_action_api.py` and `test_telegram_approvals.py` as its
      consumers, but six modules import it now
      (`test_host_action_decisions.py`, `test_nixos_config_change.py` and
      `test_host_digest.py` too). Replace the enumeration with "every module
      that drives the privileged path for real" so it cannot go stale again.
- [ ] `tests/test_host_mcp_server.py:11` and `:132` - both state the inspection
      parsers are "pinned against captured fixtures in `test_host_inspection.py`",
      but 20260731-171432 moved the thermal and nix-store pins into
      `tests/test_host_thermal.py` and `tests/test_host_nix_store.py`. Name all
      three files at each site.
- [ ] Confirm no other citation of a split-away test file survives: for each of
      `test_auth.py`, `test_telegram.py`, `test_host_inspection.py`,
      `test_agent_store.py`, `test_host_action_api.py`,
      `test_nixos_config_change.py`, `agent-chat-view.test.ts` and
      `host-view.test.ts`, grep the whole repo outside `tasks/` and decide each
      hit. `tasks/` is append-only history and is exempt.
- [ ] `ruff format` only the files edited, then run the gate.

## Definition of Done

- Neither stale citation survives
  (cmd: `rg -n "test_host_inspection.py" tests/test_host_mcp_server.py`).
- The `conftest.py` comment no longer enumerates consumers
  (cmd: `rg -n -A 6 "a real scufris-hostd" tests/conftest.py`).
- The canonical gate is green
  (cmd: `nix flake check`).

## Notes

- Epic 20260731-171411 closed with these findings open on purpose: they are
  MINOR/NIT, comment-only, and did not justify holding the branch.
- R1.1 from the same round - the out-of-scope `ruff format` pass on
  `scufris/app.py` in commit `3fe5be1` - is deliberately NOT in scope. It is
  behavior-preserving and already landed; 20260729-103712 owns that file.
- Lesson `a-citation-sweep-follows-the-renamed-name-not-the-edited-file` is the
  general form of this task; the third Step is that lesson applied.
