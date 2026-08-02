# Fix pre-existing mypy red on master (18 errors, FakeAgent/LogRecord)

- PRIORITY: 70
- TAGS: bug
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

mypy fails on master with 18 pre-existing errors in 2 files (FakeAgent
incompatible with Agent protocol; LogRecord.req attribute), while recent
task records claim green suites - something drifted after 20260720-144530
closed. Reproduced on clean 4f99091 during the flow-v2 adoption
(20260720-171850). Fix the types (or the fixtures) so mypy is green
before the next code change lands.
