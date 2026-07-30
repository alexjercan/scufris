# Retro: Add the host operator agent and the approval decision core

- TASK: 20260729-125040
- BRANCH: feat/host-operator-agent
- REVIEW ROUNDS: 2 (5 findings: 1 MAJOR, 3 MINOR, 1 NIT)

What changed and why is in `NOTES.md`; the forks are in `DECISION.md`. This is
process only.

## What went well

- **Putting the four forks to the operator before writing anything.** Two of them
  (who holds the mutating tools, what counts as the Telegram credential) would have
  been silently resolved the wrong way by inference, and the tool-audience answer
  changed the shape of half the diff. The re-cut into three tasks came out of the
  same conversation and made this task finishable.
- **Running the thing rather than reasoning about it.** Both `examples/host_agent.py`
  paths were executed and READ, which is how the "service control - reversible"
  label was caught sitting next to an undo line saying the opposite. The VM test was
  run for real (`nix build .#hostd-vm-test`, exit 0), so the new root-facing verb is
  proven on a real socket rather than only against the injected engine.
- **Probing the review finding instead of asserting it.** R1.1 was written up from a
  throwaway test whose output is quoted in REVIEW.md. That is what made it a MAJOR
  with a fix in one pass rather than an argument.

## What went wrong

- **The confirmation rule was designed from a field's NAME.** `reversal.possible`
  read like exactly the right signal for "needs a stronger acknowledgement", and it
  is wrong for the most ordinary action on this box: restarting a running unit
  reports no undo. Root cause: the rule was written into the service and the route
  before any test exercised it, so the refutation arrived as five unrelated
  pre-existing tests going red instead of as one deliberate red test. The DoD named
  `test_one_way_action_requires_stronger_confirmation`; writing THAT first, per the
  repo's own test-first convention for `test:` proofs, would have surfaced it in a
  minute and saved a rewrite of the rule plus a correction to DECISION.md.
- **A state-keyed guard with no path that clears the state (R1.1).** The BLOCKED
  refusals were written alongside the approve and deny paths, both of which clear the
  state by resuming the agent - so "what clears this?" had an answer for every path
  that was in front of me. The path where NOBODY acts (the proposal expires) was not,
  and it locked the agent out permanently. Root cause: reasoning about a guard from
  the transitions I was writing rather than from the full set of ways the guarded
  state can end.
- **Round 1 could not use the out-of-context reviewer.** Subagents are disabled in
  this session, so the review skill's default was unavailable and the round ran
  in-session (recorded as the exception in the round header). The blind spot the
  default exists to remove was mitigated by probing rather than reading, but not
  removed: an out-of-context round on this branch would still be worth having.

## What to improve next time

- When a DoD item names a `test:` proof for a RULE (not a wiring change), write that
  test and watch it fail before wiring the rule into the caller. The rule is the
  thing most likely to be wrong about the real data.
- Before shipping a guard keyed on a state, enumerate every way that state can end -
  including "nobody ever acts" - and check each one clears it. Timeouts and
  expiries are the paths that get missed, because no code path represents them.
- Say up front when a session cannot run the out-of-context review round, so the
  user can decide whether to supply one, rather than discovering it in the round
  header.

## Action items

- [x] Lessons ledger: `rule-from-a-field-name-needs-real-data`,
      `state-keyed-guard-needs-a-clearer-on-every-path` (new), and
      `assert-a-credential-rule-with-only-that-credential` (new) appended.
- [x] Recorded the two build-time corrections in `DECISION.md` sections 5 and 6
      rather than leaving the record describing a design that was not built.
- [ ] Not created as a task: an out-of-context review round on this branch, if the
      user wants one (it needs a mechanism this session does not have).
