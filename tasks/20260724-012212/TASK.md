# Feature: export chats to markdown; for all agents

- STATUS: CLOSED
- PRIORITY: 80
- TAGS: feature

## Story

As an operator using Scufris chats, I want to export a conversation to a
readable Markdown file from any agent surface, so that I can keep or share the
agent's work without scraping the UI.

Existing context: `/export` already downloads the currently loaded browser
messages through the shared chat component, and both the orchestrator page and
per-agent detail page mount that component. This task makes export obvious,
consistent, and tested for all agent surfaces.

## Steps

- [x] Add meaningful markdown formatting for chat export: title, generated time,
      role headings, timestamps when known, and omission of empty messages.
- [x] Add a visible export action to the shared `createAgentChat` surface, while
      keeping `/export` as a slash-command path.
- [x] Wire export filenames/titles through both orchestrator and per-agent chat
      entries, so downloads are labeled by agent instead of always
      `scufris-chat.md`.
- [x] Add frontend tests proving markdown formatting, empty-export no-op, visible
      button export, and per-agent export wiring.
- [x] Run the relevant frontend checks plus repository checks touched by the task.

## Definition of Done

- Exported markdown includes a stable title, generated timestamp, role headings,
  message text, and message timestamps when present.
  (test: `chatMarkdown renders titled markdown with timestamps`)
- Empty chats do not trigger a download.
  (test: `downloadChatMarkdown does not create a blob for an empty export`)
- The shared chat UI exposes an export button and it downloads the loaded
  transcript for any mounted agent.
  (test: `exports the loaded transcript from the visible export button`)
- The per-agent page passes an agent-specific title/filename into the shared
  export control.
  (test: `startAgentChat uses an agent-specific export label`)
- Frontend and Python checks touched by the change are green.
  (cmd: `npm run ci` in `web/`; cmd: `python -m pytest tests/test_app.py
  tests/test_backends.py tests/test_sessions.py`; cmd: `ruff check .`; cmd:
  `mypy .`)

## Notes

- File pointers: `web/src/chat-commands.ts`, `web/src/agent-chat-view.ts`,
  `web/src/agent-view.ts`, `web/src/style.css`, and the matching Vitest suites.
- Keep export client-side for this task: all chat surfaces already load a
  normalized transcript through the backend protocol, so this avoids adding
  backend endpoints that would duplicate the existing read paths.

## Close-out

- Changed `chatMarkdown` and `downloadChatMarkdown` to produce titled markdown
  with generated/export timestamps, role sections, message timestamps when
  available, empty-message omission, and caller-provided filenames.
- Added a visible shared `Export` action to `createAgentChat` while keeping the
  existing `/export` slash-command path on the same `exportChat` control.
- Passed agent-specific export labels through the orchestrator and per-agent
  chat entries so the resulting downloads are named by surface.
- Added Vitest coverage for formatted markdown, empty-export no-op behavior, the
  visible export button, and per-agent filename/title wiring.
- Chose the client-side export path over a backend endpoint because each chat
  surface already owns a normalized loaded transcript; a backend export endpoint
  would duplicate transcript read paths without improving the user workflow.
- Difficulty encountered: the named sprout branch already existed with
  in-progress edits for this task. I inspected that worktree, preserved its
  changes, and completed verification on the existing branch instead of
  recreating it.
- Verification:
  - `npm run format`
  - `npm run test -- chat-commands.test.ts agent-chat-view.test.ts`
  - `npm run ci`
  - `nix develop -c bash -c 'python -m pytest tests/test_app.py tests/test_backends.py tests/test_sessions.py'`
  - `nix develop -c bash -c 'ruff check .'`
  - `nix develop -c bash -c 'mypy .'`
  - `git diff -- web/src/chat-commands.ts web/src/chat-commands.test.ts web/src/agent-chat-view.ts web/src/agent-chat-view.test.ts web/src/agent-view.ts web/src/style.css tasks/20260724-012212/TASK.md | grep -nP '[^\x00-\x7F]' || true`
- Self-reflection: when resuming an existing sprout branch, inspect and verify
  the inherited diff before editing. That preserved prior task work and made
  the remaining work a focused close-out instead of a duplicate implementation.
