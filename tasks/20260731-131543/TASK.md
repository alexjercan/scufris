# Condense repository agent guidance

- PRIORITY: 0
- TAGS: backlog, docs
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As a repository contributor, I want concise agent guidance, so that critical rules are easy to find and authoritative detail is not duplicated.

## Steps

- [x] Replace verbose AGENTS.md prose with concise commands, workflow, testing, and invariant summaries.
- [x] Move the release procedure to a dedicated durable document and update its pointer.
- [x] Verify links, ASCII punctuation, record conformance, and the documentation diff.

## Definition of Done

- AGENTS.md keeps essential repository rules and points domain detail to authoritative docs. (manual: compare retained rules and pointers against the previous AGENTS.md)
- AGENTS.md follows the global concise writing rules and contains the five Agent workflow pointers. (cmd: test "$(rg -c '^- (Tracker and epics|Examples and retention|Domain docs|Research and network|Checks and records):' AGENTS.md)" -eq 5)
- Documentation links and task records are valid. (cmd: test -f docs/RELEASING.md && rg -n 'docs/RELEASING.md' AGENTS.md README.md)
- This task's records remain conformant. (cmd: tatr check 20260731-131543)

## Notes

- Preserve security boundaries as short invariants.
- Task records remain historical sources, not live documentation.
- Full `tatr check --ledger LESSONS.md` remains red on pre-existing pending promotion decisions; user disposition required.
