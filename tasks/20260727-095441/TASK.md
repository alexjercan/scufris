# Dedupe tool-call chips in assistant meta line

- PRIORITY: 0
- TAGS: web,backlog,ui,chat
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: -

## Problem

An assistant turn's meta line lists one chip per tool call, in call order. A
turn that polls (`agent_status`, `pending_agents`) repeatedly renders the same
tool name many times in a row, e.g.

```
ran list_projects list_agents create_agent run_agent agent_status
agent_status agent_status agent_status pending_agents agent_status ...
```

The user wants only the DISTINCT tool names, each shown once, in
first-occurrence order:

```
ran list_projects list_agents create_agent run_agent agent_status pending_agents
```

Scope is deliberately narrow ("first fix only this part"): dedupe the displayed
tool-name list. No counts, no other chat/meta changes.

## Definition of Done

1. `messageMeta` renders each distinct tool name at most once, preserving
   first-occurrence order. (test: `web` vitest case asserting a reply with
   repeated tools yields one chip per distinct name)
2. The live streaming status line (`paintStatus`, the `ran ...` suffix) shows
   the same distinct-only list, so live and settled views agree.
3. `cmd: cd web && npm run ci` is green (prettier, eslint, vitest, build).

## Steps

- [x] Add a small order-preserving dedupe of tool names in `agent-chat-view.ts`.
- [x] Use it in `messageMeta` (chips) and in the live `paintStatus` `ran` list.
- [x] Add/extend a vitest case in `agent-chat-view.test.ts` for repeated tools.
- [x] Run `npm run ci` in `web/` and make it green.
