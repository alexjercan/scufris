# Make the mock backend stateful for deterministic browser QA

- PRIORITY: 68
- TAGS: feature, v0.2.0, testing, backend, agents
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Story

As a frontend tester and offline evaluator, I want the mock backend to behave
like a small persistent agent, so that reload, history, streaming, cancellation,
tool events, and error paths can be tested without a provider subscription.

## Steps

- [ ] Add failing integration coverage showing that a completed mock turn is
      currently absent after app reconstruction or transcript reload.
- [ ] Persist mock sessions and transcript messages in the configured isolated
      state directory using the same lifecycle contract as real backends.
- [ ] Add deterministic scenarios for normal streaming, delayed streaming,
      tool-call events, cancellation, and a controlled backend error.
- [ ] Add deterministic multi-agent scenarios that browser/API fixtures can
      pair with the existing `request_input` and `report_back` endpoints:
      concurrent completion, waiting for input, reported completion, and error.
- [ ] Expose scenario selection only through explicit test/demo configuration;
      keep production defaults unsurprising.
- [ ] Verify concurrent mock agents do not mix sessions or transcripts.
- [ ] Update the offline demo documentation and add a small end-to-end example
      that sends a turn, reconstructs the app, and reads it back.

## Definition of Done

- A mock reply and session transcript survive app reconstruction
  (test: `test_mock_transcript_survives_app_restart`).
- Delay, tool event, cancellation, and error scenarios are deterministic
  (test: `test_mock_backend_scripted_scenarios`).
- Concurrent, waiting, reported, and failed multi-agent scripts are
  deterministic and retain their actor/session attribution
  (test: `test_mock_backend_scripted_multi_agent_scenarios`).
- Concurrent mock agents retain separate histories
  (test: `test_mock_agent_transcripts_are_isolated`).
- The offline example succeeds (cmd: `python examples/mock_agent_roundtrip.py`).

## Notes

- Epic: 20260729-102149.
- Depends on: 20260729-102147.
- Use the persistence implementation from 20260729-102147 rather than creating
  another independent state mechanism.
- This task is a prerequisite for stable browser tests.
- V0.2.0 readiness role: provide subscription-free, restart-safe fixtures for
  the actor-aware orchestration spike and its future implementation epic.
