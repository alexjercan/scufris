# Agent page: context breakdown + weekly-usage panel

- STATUS: OPEN
- PRIORITY: 20
- TAGS: feature, agent, ui, spike

## Goal

Two read-only panels on the agent page, fed by the backend:

1. A `/context`-style view for the current session: context-window size, used %
   (a bar), cached-vs-fresh input tokens, cumulative output/reasoning, turn count
   and per-tool call counts.
2. A weekly-usage meter (in the sidebar): `rate_limits.primary.used_percent` over
   the 10080-minute (weekly) window, `resets_at` ("resets in 2d 5h"), plan type,
   and the secondary window if present.

Be honest about the limit the spike found: codex does NOT expose a per-component
context breakdown (system/tools/MCP/messages token split), so show the real axes
it does give, not a faked breakdown.

Likely surface (for `/plan`): consume `GET /api/agent/context` +
`GET /api/agent/usage`; pure render helpers + jsdom tests; theme with the
existing card/bar styles.

## Notes

- Spike: tasks/20260719-212152/SPIKE.md.
- Depends on tatr 20260719-212203 (context + usage endpoints).
- Usage refreshes only when a turn runs (label "as of last turn"); do not force a
  refresh turn. Keep render side-effect-free for jsdom; escape host strings.
