# Retro: BC5 end-to-end example + acceptance test

- TASK: 20260723-094318
- DATE: 20260723
- OUTCOME: landed, 1 review round (APPROVE, zero findings)

## What we set out to do

Prove the bidirectional agent<->orchestrator comms channel (BC1-BC4) self-heals
the exact stalled-merge case the spike exists to fix - a runnable example plus a
harness-level acceptance test covering BOTH wake paths.

## What went well

- Mapped the whole machinery up front (a dedicated Explore pass over the mock
  backend, request_input/outcome store, pending/acknowledge, message_agent/chat
  resume, and the auto_wake bridge) BEFORE writing a line. That map is what made
  the test faithful and caught the load-bearing subtlety below.
- Found the key semantic BEFORE it caused a flaky test: `mark_finished` preserves
  a WAITING outcome only when the finishing run's id MATCHES the run that set it
  (agent_store.py:578-585). The resume via /chat is a NEW run, so its DONE
  overwrites WAITING - i.e. answering clears the signal by itself, and acknowledge
  is idempotent cleanup. Designing the assertions around this (assert `pending ==
  []`, never assert acknowledge's bool) made both the test and the example
  race-free across every interleaving.
- Reused the existing BC4 async-httpx + ASGITransport + blocking-stream template
  (test_app.py) rather than inventing a harness, then extended it past "was the
  orchestrator woken" to the full answer-by-resume + clear.
- Scripted the fake backend to block ONLY the first sub-agent, non-orchestrator
  turn (`blocked_once` + `release`), so the same stream faithfully holds the run
  in-flight yet lets the wake turn and the resume turn complete - no deadlock.
- Stress-ran the async test 6x (12 parametrized executions) and the example 3x
  before review; the reviewer independently re-ran 5x. Zero failures.

## What went wrong / friction

- The example's first draft asserted `acknowledged is True`, which PASSED by
  timing luck: with no prior /run, the request_input WAITING has run_id="" so the
  resume's DONE overwrites it, and acknowledge would return False once that landed.
  Caught by reasoning through mark_finished's overwrite semantics (not by a failing
  run - it was green). Fixed to assert only `pending == []`, which holds under
  every interleaving. Lesson: a green test that encodes a race is still wrong; a
  passing assertion whose truth depends on which callback lands first is a bug.
- Standing in the MCP tools with their HTTP endpoints is the right call for a
  mock-backend acceptance (the mock cannot run real MCP tools), but it means the
  test proves the ENDPOINT contracts, not codex actually choosing to call the
  tools - that half is the SC1/SC2 steering plus the still-pending live probe.
  Documented this honestly in the test/example docstrings so they do not
  over-claim.

## Lessons (candidates for the ledger)

- `acceptance-assert-the-end-state-not-the-cleanup-return`: when a loop can be
  resolved by more than one mechanism (here: the resume's DONE overwriting WAITING
  OR an explicit acknowledge), assert the OBSERVABLE END STATE (`pending == []`),
  not the return value of one of the mechanisms. Asserting `acknowledged is True`
  encodes a race on which callback lands first - green by luck, wrong in principle.
- `mark_finished-preserves-waiting-only-within-the-same-run`: a WAITING outcome is
  kept through turn-end only when the finishing run's id equals the run that set it
  (agent_store.py:578-585); a later/other run's terminal state overwrites it. So a
  message_agent resume (a new run) naturally clears the sub-agent from
  `pending_agents` when it finishes - answering IS the clear; acknowledge is
  belt-and-suspenders.

## Deferred to Finish / follow-ups

- Manual live-probe (SC1/SC2, still pending): this acceptance covers the loop
  against a FAKED backend and the HTTP contracts; it does NOT prove a real codex
  sub-agent chooses to call request_input, nor the steered orchestrator chooses to
  poll/answer. One real-backend probe still closes that last gap for the whole
  comms arc.
