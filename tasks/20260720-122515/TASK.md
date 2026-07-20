# Agent chat: slash-command palette in the composer

- STATUS: OPEN
- PRIORITY: 40
- TAGS: feature,agent,ui

## Goal

Add a `/command` palette to the chat composer (like Claude Code's slash commands).
codex exec/app-server has NO slash-command mechanism (it is a TUI-only feature), so
this is a client-side feature: intercept a leading `/` in the composer, show an
autocomplete menu, and on select either expand to a prompt template or call an
action/API directly.

## Notes

- Spike: tasks/20260720-122301/SPIKE.md.
- User: "I would also like to be able to have /commands like in claude code ...
  like using skills."
- Candidate commands: `/new` (new chat), `/settings` (nav), `/today` /`/tasks`
  (prompt or direct den-tool call), `/export` (download session md), `/help`
  (list commands). Some map to existing APIs/actions; some expand to a steering
  prompt. Design an extensible registry {name, description, run}.
- UI: an autocomplete menu above the composer filtered as the user types `/`;
  Enter/Tab to accept; Esc to dismiss; keyboard-navigable (a11y).
- Frontend-only (may add a tiny export endpoint). Keep render side-effect-free for
  jsdom; escape everything. Note in the doc: codex SKILL.md skills are a separate,
  more experimental capability mechanism - not this task.
