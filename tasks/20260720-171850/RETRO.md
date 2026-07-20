# Retro: Adopt flow v2 (scufris)

- TASK: 20260720-171850
- BRANCH: chore/flow-v2-adoption (landed as 43e8a87 via sprout land)
- REVIEW ROUNDS: 1 (out-of-context APPROVE, 1 NIT taken)

## What went well

- The largest history-normalization of the six repos (36 files) survived
  an exhaustive review - every changed line read, not sampled - because
  the work agent verified each token was a severity use before replacing
  and kept residue honest rather than forcing ticks.
- The residue model worked as designed: 5 unticked boxes with reasons,
  now awaiting user rulings at the goal's Finish instead of being silently
  ticked or blocking the migration.

## What went wrong

- mypy is red on master (18 pre-existing errors) while recent task records
  claim green suites - drift between records and reality that the
  migration surfaced but could not explain. Filed as its own bug task.

## Action items

- [x] Filed the mypy bug task; residue forwarded to the umbrella GOAL.md.
