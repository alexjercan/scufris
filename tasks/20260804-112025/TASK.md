# make _wait_state's agent-run polls deterministic instead of a 2s timeout

- PRIORITY: 40
- TAGS: bug,tests,flaky
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

`tests/test_app.py:2577` `_wait_state` polls `/api/agents/{id}/status` 200 times
at 10ms and then RETURNS whatever it last saw rather than failing, so a slow
machine turns a real wait into a silent wrong-state assertion downstream.

Observed once during review of 20260803-213242: a full-suite run failed
`tests/test_app.py::test_orchestrator_chat_uses_server_cwd` and the same test
passed in isolation and on the next full run (1109 passed, 1 skipped).

Pre-existing; not caused by the workspace carve. Filed from that review rather
than fixed in it.

Wanted: either wait on a real signal from the portal loop instead of sleeping,
or raise the budget and make the helper FAIL on timeout so a timing loss reads
as a timeout rather than as a state mismatch.
