# Retro: A2 AgentBackend interface + CodexBackend + status + probe

- TASK: 20260720-221935
- BRANCH: feature/agent-backend (landed 4d6850a)
- REVIEW ROUNDS: 1 out-of-context APPROVE (1 NIT, accepted-not-taken)

## What went well

- Building the interface as a thin adapter over the already-reviewed agent.py
  runners + sessions.py readers kept the diff small and the risk low; read_status
  is pure reuse of read_context/read_transcript.
- Testing stream delegation by monkeypatching the runners (asserting the exact
  forwarded arg tuple + a `fail_exec` guard on mode selection) proved the adapter
  logic precisely without a slow fake-codex subprocess - the subprocess cwd
  wiring was already proven at the runner level in A0, so re-proving it here
  would have been redundant.
- The probe was cheap (7s) and high-value: it caught a design error before A3.

## What went wrong

- The spike (which I wrote) generalized "the agent runs `/flow`" across backends.
  The live probe showed that is wrong: codex is already an autonomous agent and
  `/flow` is a Claude-Code-only skill. Root cause: reasoning about codex's
  behavior from the Claude-Code mental model instead of running it. This is the
  third occurrence of `probe-runtime-on-target-host-early` - now promoted to the
  ledger's Pending promotions for a spike-skill rule.

## What to improve next time

- When a spike generalizes a capability across two tools ("both backends will do
  X"), treat the shared-capability claim as a hypothesis and probe the
  less-familiar tool before planning against it. The interface abstraction was
  fine; the ASSUMPTION about what flows through it was not.

## Action items

- [x] NIT reviewed and consciously deferred (read_status materialization).
- [x] NOTES.md records the probe + the /flow correction for A3.
- [x] `probe-runtime-on-target-host-early` bumped to x3 and moved to Pending
  promotions (proposed target: spike/plan skill).
- A3 plan MUST hand each backend a generic goal prompt, not a hard-coded /flow.
