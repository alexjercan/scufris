# F4: per-agent chat UI on the detail page (reuse the agent-view chat helpers)

- STATUS: OPEN
- PRIORITY: 36
- TAGS: agents,frontend


## Goal

The per-agent CHAT UI on the detail page: reuse the pure chat helpers from
`agent-view.ts` (parseSseFrames, sendChatStream, markdown render, no-yank scroll,
composer) de-globalized into `agent-detail.ts`'s own state, targeting the
per-agent chat/events/transcript endpoints. Multi-turn conversation with the
agent, like the landing page.

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (recommendation F4; reuse map - the chat components lift, the
  wiring is new).
- Depends on: 20260721-112433 (F1), 20260721-112435 (F3), 20260721-112436 (B4).
