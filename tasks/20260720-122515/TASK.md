# Agent chat: slash-command palette in the composer

- STATUS: CLOSED
- PRIORITY: 40
- TAGS: feature,agent,ui
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

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

## Implementation (frontend-only, no backend)

- `index.html`: a `#chat-palette` (role=listbox) inside `.chat__form`, above the
  composer; placeholder now hints "type / for commands".
- `agent-view.ts`: an extensible `SLASH_COMMANDS` registry `{name, description,
  run}` with six commands - `/new` (new chat), `/settings` (nav to /settings/),
  `/tasks` and `/host` (fill a prompt the user then sends), `/export` (download the
  chat as markdown, client-side Blob), `/help` (list commands as a system line).
  Pure exported helpers `matchSlashCommands(value)` (matches a lone `/token` at the
  start, no space) and `chatMarkdown(messages)` (for /export). Palette UI wired in
  `initChat`: opens/filters on input; ArrowUp/Down move the selection, Enter/Tab
  accept, Escape dismisses, mousedown on an item runs it; the keydown handler
  checks the palette FIRST so Enter accepts a command instead of sending; blur and
  submit close it. escapeHtml on names/descriptions.
- `style.css`: `.chat__form { position: relative }` + a floating `.chat__palette*`
  (above the composer, active-row highlight).

## Tests / verification

- `agent-view.test.ts`: `matchSlashCommands` (bare `/` lists all, prefix filters,
  a space ends matching, unknown -> none); `chatMarkdown`; and palette interaction
  via `initChat` - opens/filters on typing, Enter runs the highlighted command
  (fills a prompt, sends nothing), arrows move selection, Escape closes, and a
  normal message still sends when the palette is closed. 91 frontend tests green;
  built `dist/index.html` ships `#chat-palette` and all six commands.
- codex SKILL.md skills remain a separate, deferred mechanism (not this task).
