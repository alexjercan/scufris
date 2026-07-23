# Goal: correct + transparent agent tool surface, and claude tool use

- DATE: 20260723
- UMBRELLA TASK: 20260723-192825
- LANDING SCOPE: squash-merge each task to master (local), no push. Standard flow.

## Goal

Make the agent tool surface CORRECT and TRANSPARENT, and extend tool use to the
claude backend. Three threads, from a live review of a running app:

1. Docs drift: README says agents are "off by default" and tells you to
   `export SCUFRIS_AGENT_ENABLED=1`, but the code default is `agent_enabled=True`.
   Reconcile so the docs match reality (or the default matches the docs).
2. Wrong + opaque per-agent tools: a codex SUB-AGENT's UI shows "18 tools
   available" - the ORCHESTRATOR's full role surface - when a sub-agent is
   role-scoped to ONLY `request_input`. Fix the count/list to be role-correct, and
   add an "available tools" panel to EACH agent's settings page (mirroring the
   orchestrator settings) so an agent's real tool surface is transparent.
3. Claude tool-use parity: claude-backed agents get no scufris MCP tools (the
   `request_input` callback, and for the orchestrator the control tools) - codex
   only. Figure out how to close the gap (a spike, since "figure out a way" is
   still open), then implement.

## Done means

1. README no longer misstates the `agent_enabled` default; docs and code agree.
   (cmd: `grep -n "off by default" README.md` reflects the true default)
2. A codex sub-agent's tools listing shows only its role-scoped tools
   (`request_input`), not the orchestrator's full set. (test: a role-scoped tools
   endpoint/count returns the agent set for a sub-agent) (manual: the agent page
   shows 1 tool, not 18)
3. Each agent's settings page renders its available-tools panel like the
   orchestrator's. (manual: open a sub-agent's settings, see its tools)
4. A claude-backed agent can use the `request_input` tool (loop parity with
   codex), or the spike documents precisely why not and what it needs. (test/manual
   per the spike outcome)

Overall: `ruff check .`, `mypy`, `python -m pytest` green on master; the full
comms loop still self-heals (BC5 acceptance stays green).

## Tasks

Ordered by priority; ticked as each lands.

- [x] 20260723-193159 (p42, docs) Reconcile agent_enabled default vs README off-by-default drift
      landed 3f3af48; trivial-diff self-review (no round); .env.example was already correct
- [ ] 20260723-193216 (p40, backend+ui) Role-scoped per-agent tools endpoint + tools panel on each agent's settings
- [ ] 20260723-193218 (p30, spike) Give claude-backed agents scufris MCP tools (request_input parity) - seeds impl task(s)

## Decisions (load-bearing, architectural)

## Manual acceptance (batched for the user at Finish)
