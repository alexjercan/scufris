# Split the oversized test suites under the size cap

- STATUS: OPEN
- PRIORITY: 70
- TAGS: refactor, v0.2.0, testing, maintainability
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT
- PARENT: 20260731-171411
- DEPENDS ON: 20260731-171420

## Story

As a maintainer, I want oversized test suites split by domain, so that working
on one area loads only that area's tests.

## Steps

- [ ] Split `tests/test_telegram.py` (1760), `tests/test_host_action_api.py`
      (1285), `tests/test_auth.py` (1219), `tests/test_host_inspection.py`
      (1076), `tests/test_nixos_config_change.py` (1044),
      `tests/test_agent_store.py` (937), and
      `web/src/agent-chat-view.test.ts` (1183) / `host-view.test.ts` (997) by
      the behavior under test.
- [ ] Move shared setup into existing fixtures/conftest rather than copying it
      into each new file.
- [ ] Preserve every assertion; a split that drops or weakens a test is a
      regression.
- [ ] Apply the epic comment policy to every file touched.
- [ ] Remove the corresponding allowlist entries from the size guard.

## Definition of Done

- No file under `tests/` or `web/src/**.test.ts` exceeds 900 lines, except
  `tests/test_app.py` (owned by 20260729-103712)
  (cmd: `python scripts/check_file_size.py`).
- Test count is unchanged or higher before and after
  (cmd: `python -m pytest --collect-only -q | tail -1`).
- Both canonical gates pass
  (cmd: `nix flake check && cd web && npm run ci`).

## Notes

- Epic: 20260731-171411.
- Depends on: 20260731-171420. Run after the source splits so tests move once, not twice.
- Splitting by fixture instead of by behavior produces the same context cost;
  split by what is being tested.
