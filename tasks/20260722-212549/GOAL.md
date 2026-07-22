# Goal: interactive "try it" tool runner on the Settings page

- DATE: 20260722
- UMBRELLA TASK: 20260722-212549
- LANDING SCOPE: squash-merge each task to the local default branch (master)
  via `sprout land`; do NOT push (pushing is the user's call).

## Goal

Give the homelab operator an interactive "try it" runner on the Settings page:
click a tool card, get a form generated from that tool's `inputSchema.properties`,
hit Run behind a confirm step, and see the tool's result rendered inline (JSON or
text) - WITHOUT going through a chat turn / the agent. This lets an operator debug a
single scufris MCP tool in isolation ("does host_stats work right now?").

This is the interactive item the operator-console spike (tasks/20260720-134459)
deliberately deferred out of task 20260720-122517 (now CLOSED) because it needs a
real run-a-tool capability, not just read-only rendering.

Consent model (decided with the user 20260722): a UI confirm step before every run;
the endpoint refuses any tool in `disabled_tools`; NO new gating setting (the tool
set is already curated - fixed flags, bounded output, no arbitrary-command tool;
most tools are read-only, only `tatr_new` writes and it is bounded to tatr).

## Done means

1. A new backend endpoint runs exactly ONE scufris MCP tool by name with the given
   args, in-process via the FastMCP server, bypassing codex/the agent, and returns
   the tool's result. (test: pytest hits the route with a FastAPI TestClient and
   gets host_stats/tatr_ls output back)
2. The endpoint is safe on bad input: an unknown tool name and malformed/oversized
   args return a controlled 4xx (not an uncontrolled 500); a tool in
   `disabled_tools` is refused (403). (test: pytest cases for each)
3. The Settings page renders a form from a tool's `inputSchema.properties`, gates
   Run behind a confirm step, and renders the returned result (JSON/text) with all
   values escaped. (test: frontend test for form-gen + escaping; manual: click a
   tool, run it, see the result)
4. Docs/changelog updated where the change touches them (CHANGELOG entry; any
   Settings/README surface that describes the page). (cmd: grep check in the task)

Overall: the full QA gate is green - `ruff check .`, `mypy .`, `python -m pytest`,
and the frontend test suite (and `nix flake check` if run).

## Tasks

Updated as tasks land (one line per land, like a spike's Fix record).

- [ ] 20260720-134545 (p21, backend) run-one-tool endpoint + param schema contract
- [ ] 20260722-213000 (p20, frontend) 'try it' runner UI (form + confirm + result);
      depends on 20260720-134545

## Manual acceptance (batched for the user at Finish)

Accumulates `manual:` DoD items as tasks land; presented at Finish.

- (pending) whole-flow: on the running app, open Settings, click a tool card, fill
  the generated form, confirm, Run, and see the correct result rendered inline with
  no chat turn.
