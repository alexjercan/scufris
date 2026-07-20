# Review: Fix pre-existing mypy red on master (FakeAgent/LogRecord)

- TASK: 20260720-174021
- BRANCH: bug/mypy-green

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (fresh subagent; in-session pass re-ran the suite
  and re-verified the StreamEvent-membership claim)

No blocking or major findings. mypy went from 18 errors to 0; ruff clean;
pytest 66 tests green. No `type: ignore` added, no tests weakened or deleted.

- [x] R1.1 (verified, not a finding) tests/test_app.py - `chat_stream` return
  type changed from `AsyncIterator[object]` to `AsyncIterator[StreamEvent]` is
  HONEST: FakeAgent yields only `StreamTool` and `StreamDone`, both genuine
  members of the `StreamEvent` union (scufris/agent.py:94-96). No protocol
  drift masked.
- [x] R1.2 (verified) FakeAgent is passed to `create_app(agent: Agent | None)`
  at 11 call sites, so mypy structurally checks it against the full `Agent`
  protocol; mypy passing proves it implements all protocol methods with
  correct signatures - the annotation hides no missing/mismatched method.
- [x] R1.3 (verified) tests/test_app.py - `bool(image_paths and
  os.path.isfile(image_paths[0]))` is logically equivalent to the original,
  narrows `image_paths` away from `None` before indexing, and the
  `image_existed is True` assertion is not weakened.
- [x] R1.4 (verified) tests/test_logsetup.py - `record.__dict__["req"]` reads
  the dynamically-attached attribute type-honestly (getattr with a constant
  trips ruff B009); still fails (KeyError) if the filter fails to attach
  `req`, so the assertion is not weakened.

No open `manual:` DoD items.
