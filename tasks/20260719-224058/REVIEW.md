# Review: agent tatr tools - create task, sort, filter-language docs

- DATE: 20260719
- VERDICT: APPROVE (1 round)

## Scope reviewed

`scufris/mcp_server.py` (`tatr_new`, `sort` on `tatr_ls`, filter docs, updated
server docstring), `tests/test_mcp_server.py`.

## Correctness

- Live-verified against real `tatr`: `tatr_new("...", priority=5, tags="feature,
  agent")` created a task with the priority/tags applied; `tatr_ls(sort="priority")`
  ordered 9 before 5; `tatr_ls(filter=":tags contains bug")` returned only the
  bug; a bad sort returned the guard message; the `tatr_ls` description carries
  the operator docs (`contains` present), and `tatr_new` is registered.
- `tatr_new` is the server's first write tool; the module docstring was updated
  from "read-only" to name it explicitly and state why the write is safe (separate
  trusted process, fixed flags, bounded to tatr's tasks dir). It validates: title
  required (stripped), priority non-negative, tags cleaned to a comma list. Title/
  tags/filter all go through the fixed-arg `_run` (no shell string), so a hostile
  title or filter cannot inject a command - it is just a tatr argument.
- The filter language is documented IN the `tatr_ls` tool description (fields /
  operators / connectives / examples) so the model can actually use `-f`, which is
  the point of the task.
- Second-resolution ID collisions are handled honestly: the tool surfaces tatr's
  "already exists" text (docstring tells the caller to retry), and the one test
  that creates two tasks spaces them with `sleep(1.1)` so it does not flake.
- `sort` is validated against the allowed set and returns a usable error string
  (consistent with the tool convention of returning text, not raising).
- Full suite green: `ruff`/`ruff format`/`mypy` (10 files)/`pytest`. The
  `/api/agent/tools` subset test still holds (tatr_new is additive).

## Nits (non-blocking)

- `tatr_new` writes into the tatr `tasks/` dir of the MCP server's cwd (the repo
  when served from root) - i.e. the agent can add to the project backlog. That is
  the intended product behavior ("the agent runs custom tatr commands"), not a
  leak; worth being aware of.
- A `tatr_edit` companion (move status/priority) was considered and left out - the
  ask was create + sort + filter docs; easy to add later if wanted.

## Verdict

APPROVE. The agent gained real tatr reach: create tasks, sort listings, and a
documented filter language, with the one write tool curated and bounded and the
security contract (fixed args, timeout, bounded output) intact. Live-verified.
