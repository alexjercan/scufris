# Review: read-only settings/config page

- VERDICT: APPROVE
- ROUND: 1

## Summary

A new `/settings/` nav page shows the agent's read-only configuration (status,
backend, model, auth mode, sandbox, tools) plus MCP servers (with built-in/
configured badges) and the tools as cards, from a new `GET /api/agent/config` and
the existing `/api/agent/tools`. Reuses the multipage webpack pattern; served by
`StaticFiles(html=True)` with no backend change. 131 pytest + 79 frontend green;
verified end-to-end (serve + curl of `/settings/` and `/api/agent/config`).

## What is good

- Correct page-vs-panel call: a nav page matches the user's "settings page" ask,
  reuses the proven Agent/Stats multipage setup, and is the natural home for the
  future editable-settings work. The tools now have a real home off the chat head
  (unblocks task 102600).
- Clean separation: a side-effect-free `settings-view.ts` with a pure
  `renderSettings(root, config, tools)` and a thin `startSettings` entry - jsdom
  can drive the render without fetch (per `side-effect-free-module-for-jsdom-tests`).
- Everything is escaped; a hostile tool name/description test proves no `<img>`/
  `<script>` is injected even though config values are semi-trusted.
- Verified at all levels: unit (render branches), API (config aggregation +
  built-in-server gating), and a real serve+curl e2e (`frontend-verify-needs-e2e-serve`).

## Findings

- FIXED in-review - the Tools section originally listed the tool catalog even when
  `tools_enabled` is false (the `/api/agent/tools` endpoint enumerates them
  regardless of the flag). On a page whose whole job is to answer "why won't it use
  a tool?", showing 6 cards next to "tools: disabled" is contradictory. Now, when
  tools are disabled, the section says "tools are disabled
  (SCUFRIS_AGENT_TOOLS_ENABLED=0)" and shows no cards; pinned with a test.
- MINOR (accepted) - MCP server display shows the id + built-in/configured source,
  not the command/args. That is the right amount for a read-only overview; the
  command paths add noise (and the built-in one is an implementation detail).

## Verdict

APPROVE. The page cleanly delivers the read-only config + nicer tools, wires
through the existing multipage/StaticFiles machinery with no special-casing, and
the one real inconsistency (catalog vs disabled) was fixed in review with a pin.
