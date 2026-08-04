# Split test_orchestrator_routers.py along its three rigs

- PRIORITY: 0
- TAGS: refactor,backlog,tests,backend
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Story

As a maintainer, I want `tests/test_orchestrator_routers.py` split along its
three rigs, so that adding a fake or a case to any one of them stops colliding
with the repo's 900-line test file cap.

## Notes

- Source: `tasks/20260803-102351/REVIEW.md` round 1 process signal and
  `tasks/20260803-102351/DECISION.md` Consequences.
- The file sits at 897 lines against the 900-line cap in
  `scripts/check_file_size.py`. Task 20260803-102351 could not add a six-line
  `fork_seed` method to the shared `FakeRunService` there and landed
  `ForkingRunService(FakeRunService)` in `tests/test_agent_run_router.py`
  instead.
- The three rigs are project, agent-run and agent-record. The
  `test_domain_routers.py` / `test_orchestrator_routers.py` /
  `test_chat_router.py` / `test_agent_run_router.py` family is the existing
  precedent for splitting by surface.
- Consequence to unwind: the run-service fake is currently split across two
  files, so a reader looking for its full surface must follow the subclass.
- The guard is a ratchet by design; allowlisting the file is out of scope.
