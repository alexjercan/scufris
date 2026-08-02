# Spike: agent page expansion (sidebar, sessions, context, usage, MCPs)

- PRIORITY: 0
- TAGS: spike, agent, ui
- KIND: SPIKE
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Explore whether the agent page can grow a claude.ai-style sidebar with session
switching, a `/context`-style per-session view, an overall weekly-usage
indicator, and more MCP servers/tools - and what the `codex exec` backend makes
possible. Deliverable is the research doc + seeded tasks.

## Outcome

RECOMMENDED. Probing `$CODEX_HOME` (codex 0.144.4) showed the data already lives
on disk: sessions are resumable JSONL rollouts, and the `token_count` event
carries the context window, token usage, and a `rate_limits` block with the
weekly (10080-min) window used%. Sessions, context %, and weekly usage are fully
feasible; the only real limit is the fine-grained per-component `/context` split
(codex does not expose it) - show its real axes instead. More MCPs are feasible
(register via `-c` like Scufris; expand the Scufris server's tools cheapest).

See tasks/20260719-212152/SPIKE.md. Seeded: 20260719-212203 (backend),
20260719-212205 (sidebar), 20260719-212207 (context/usage panel), 20260719-212208
(MCP reach).
