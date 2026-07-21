# B5d: converge the landing + per-agent chat UI on one component

- STATUS: OPEN
- PRIORITY: 31
- TAGS: agents,frontend

## Goal

Make the landing page and the per-agent detail page use ONE chat component
(`agent-chat-view.ts`, the F4 lean chat with injected deps + local state),
retiring the globals-heavy landing chat in `agent-view.ts`. The orchestrator's
page ALSO shows the session switcher (B5c) + context/usage boxes; project agents
show the lean chat only. This removes the duplicated chat implementations.

## Coarse steps (/plan expands)

- [ ] Make the landing entry mount `agent-chat-view` against the orchestrator's
      endpoints (chat/transcript) + the session switcher + context/usage sidebar
      (reuse the F5 sidebar boxes). The landing IS the orchestrator's page now.
- [ ] Extract any still-landing-only pure helpers worth keeping (markdown,
      slash commands, image attach, export) into shared modules the one chat
      component can opt into; drop the rest.
- [ ] Retire `agent-view.ts`'s duplicated chat/log/composer once the landing
      page uses the shared component. Keep the tests meaningful (port them).
- [ ] The session switcher renders ONLY for the orchestrator (multi-session);
      project agents omit it.

## Definition of Done

- One chat component serves both the orchestrator (landing) and project agents;
  `agent-view.ts` no longer holds a second chat implementation
  (cmd: `grep -rn "renderLog\|sendChatStream" web/src/agent-view.ts` -> gone/thin).
- The orchestrator page shows the session switcher; a project page does not.
- Full web gate green.
- manual: the landing chat looks/feels like before but is the shared component.

## Notes
- Depends on: B5c (20260721-180219). Blocks: B5e (partly).
- Biggest FRONTEND slice; agent-view.ts is ~1300 lines with many landing-only
  features (sessions sidebar, fork/edit, image attach, slash commands, export).
  Decide per-feature: port to shared, keep orchestrator-only, or drop.
