# Agent tatr tools: create task, sort, filter-language docs

- PRIORITY: 40
- TAGS: feature, agent, mcp, spike
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Implementation

- `mcp_server.py`: `tatr_new(title, priority=0, tags?)` (validated, fixed-arg
  `tatr new -p -t`); `tatr_ls` gained a validated `sort` (created|priority|title)
  and its docstring now documents the `-f` filter language (fields/operators/
  connectives/examples) so the model can use it; server docstring updated to name
  `tatr_new` as the one bounded write tool. Tests: registration set, create +
  reject-empty/negative, bad-sort guard, sort+filter integration (spaced for the
  second-resolution ID collision). Live-verified with real tatr.

## Goal

Give the agent real tatr task-management reach via the Scufris MCP server, so it
can do the "run custom tatr commands" job the product is built around:

1. `tatr_new(title, priority?, tags?)` - create a task (`tatr new "..." -p N -t
   a,b`). This is the MCP server's FIRST write tool; keep it curated/bounded
   (fixed flags, no arbitrary args), and update the server docstring which today
   says "read-only".
2. Add a `sort` argument to `tatr_ls` (`created` | `priority` | `title`), passed
   through as `-s`.
3. Document the filter query language IN the `tatr_ls` tool description so the
   model actually knows how to use `-f`: fields `:status` / `:priority` /
   `:tags`; operators `eq` / `contains` / `in [...]`; connectives `and` / `or` /
   `not` with parens; e.g. `(:status eq OPEN) and (:tags contains feature)`.

## Notes

- Spike: (none - direct feature request). tatr surface from the tatr SKILL.md:
  `tatr new "Title" [-p pri] [-t tags]`, `tatr ls [-s created|priority|title]
  [-f '<query>']`, second-resolution IDs (a same-second `new` fails - surface the
  error text rather than retrying inside the tool).
- SECURITY: `tatr_new` writes files, but the MCP server runs as a separate trusted
  process spawned by codex (not under the model's read-only file sandbox), and the
  write is bounded to tatr's own `tasks/` dir via fixed args - no arbitrary paths,
  no shell string. Keep the `_run` contract (fixed arg list, timeout, bounded).
- Validate/normalize inputs: priority a non-negative int; tags a simple comma
  list; title required. Consider a `tatr_edit` companion (status/priority) only if
  cheap; the ask is create + sort + filter docs.
- Update `test_mcp_server.py` (tool set now includes `tatr_new`; `tatr_new` +
  sorted/filtered `tatr_ls` exercised against a temp tasks dir like the existing
  tatr tests).
