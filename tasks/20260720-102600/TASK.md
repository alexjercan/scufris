# Agent chat: redesign the chat head (drop the duplicate Agent title, slim it)

- STATUS: OPEN
- PRIORITY: 30
- TAGS: feature,agent,ui

## Goal

Redesign the chat pane head. Drop the redundant `<h2>Agent</h2>` title (the nav
already has an active "Agent" link), and slim the head to at-a-glance actionable
info (model, context fill / readiness). Remove the raw "tools" toggle from the
head; its entry point moves to the settings/config view (see the settings task).
Pure frontend.

## Notes

- Spike: tasks/20260720-102348/SPIKE.md.
- User feedback: dislikes "Agent / model gpt-5.5 / TOOLS" at the top of the chat
  window and wants the tools moved elsewhere.
- Coupled with the settings/tools task: plan that task's tool presentation before
  finalizing where the tools entry point lands.
- Files: `index.html` chat__head/agent-bar, `agent-view.ts` renderAgentPanel, `style.css`.
