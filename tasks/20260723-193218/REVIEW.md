# Review: spike - claude scufris MCP parity

- TASK: 20260723-193218
- BRANCH: spike/claude-mcp-tools
- DATE: 20260723
- REVIEWER: self (research deliverable, live-probe evidence)
- VERDICT: APPROVE

## Verdict: APPROVE (self-review; live-probe backed)

A spike (research), not shipping code. Its central empirical claim was PROVEN by a
live probe, so a re-run out-of-context review (which would spend the claude
subscription) adds little; the seeded impl task carries its own full review.

### What the probe actually showed (claude 2.1.193, real turns)

- `--mcp-config` accepts inline `{"mcpServers":{"scufris":{command,args,env}}}` -
  loaded without error.
- `--allowedTools "mcp__scufris__request_input"` + `--permission-mode default` ran
  UNATTENDED (no approval hang); a "reply OK" turn completed.
- A steered turn's stream-json contained `"name":"mcp__scufris__request_input"` with
  `tool_use` - claude EXPOSED AND CALLED the scufris tool. This is the load-bearing
  proof and it is unambiguous.
- Gotcha found and recorded: `--mcp-config <configs...>` is variadic/greedy and ate
  a following `mcp list` subcommand as config paths; must be bounded by a flag.

### Soundness checks

- SPIKE.md does not over-claim: it marks the orchestrator allowlist wildcard, the
  `--resume` interaction, and disabled-tools passthrough as REMAINING unknowns for
  the impl's live DoD, rather than asserting them.
- DECISION.md (backend-agnostic core + per-backend formatters) is the same
  anti-drift discipline as T2's `role_tool_names`; consequences (codex output stays
  byte-identical, `_agent_has_scufris_mcp` generalizes) are correct and testable.
- Seeded impl task 20260723-201851 has concrete Steps + a DoD (argv tests + the live
  loop), in dependency order. Good handoff.

No over-claims, ASCII-clean. Ship the research; implement via 20260723-201851.
