# Retro: codex agent in auto/edit permission mode still runs read-only

- TASK: 20260721-183828
- BRANCH: fix/codex-permission-sandbox
- REVIEW ROUNDS: 1 (APPROVE, out-of-context, no findings)

## What went well

- Diagnostic-first paid off hard. The filed task carried a plausible theory
  (missing `approval_policy=never`, like the exec MCP path needs). Three live
  probes against real codex 2.x - not reasoning - RULED THAT OUT (probe #2:
  danger-full-access ran a shell command fine with the default approval
  policy) and pinned the real cause (probe #3: a two-turn run wrote on turn 1
  and failed on the resume). Guessing would have shipped an approval-policy
  change that fixed nothing.
- The wire contract came from `codex app-server generate-ts`, not memory:
  `ThreadResumeParams` genuinely accepts `sandbox`, which is what made the
  one-line fix safe.
- The regression test is a logging fake that records the actual JSON-RPC the
  runner sends and asserts on the resume params; it KeyErrors without the fix,
  so it can actually fail.

## What went wrong

- The original app-server runner sent `thread/resume {threadId}` only. Root
  cause: it was written by analogy to a persistent server where a resumed
  thread keeps its start-time sandbox - but scufris spawns a FRESH
  `codex app-server` process per turn, so "resume" restores conversation state
  but NOT the process-level sandbox. The stateless-per-turn architecture makes
  session-scoped runtime settings turn-scoped, and that was not reasoned
  through when the resume path was first written.
- Near-miss: the existing `codex-resume-rejects-sandbox` lesson (for
  `codex exec resume`) says the exact OPPOSITE - exec resume ERRORS on a
  repeated `--sandbox`. Applying that lesson to the app-server path by analogy
  would have been wrong. The two resume mechanisms have inverse sandbox
  semantics; only reading the app-server contract + probing disambiguated it.

## What to improve next time

- When a runner spawns a fresh subprocess per turn, treat EVERY session-scoped
  runtime setting (sandbox, model, approval policy, cwd) as something the
  resume path must re-send, not something the resumed session restores. Audit
  the resume call against the start call for dropped params.
- Do not carry a same-named-verb lesson across two different transports
  (exec resume vs app-server thread/resume) without re-checking the contract;
  the names collide, the semantics do not.

## Action items

- [x] Lesson `resume-must-re-send-per-turn-runtime-settings` added to LESSONS.md
      (domain), cross-linked with `codex-resume-rejects-sandbox` as its inverse.
- Note (not blocking): the claude flavour and a per-turn sandbox change
  mid-session were deferred; file a follow-up only if observed.
