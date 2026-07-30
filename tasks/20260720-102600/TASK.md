# Agent chat: redesign the chat head (drop the duplicate Agent title, slim it)

- STATUS: CLOSED
- PRIORITY: 30
- TAGS: feature,agent,ui
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

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

## Implementation

- `index.html`: dropped `<h2 class="card__title">Agent</h2>` (redundant with the
  active "Agent" nav link) and the `.agent-bar` wrapper. The head is now a slim
  flex row: `#agent-model` (model) on the left and a compact `#agent-tools-link`
  (`<a href="/settings/">`) on the right. Removed the inline `#agent-tools` panel
  and the `#agent-tools-toggle` button entirely - the tools live on the Settings
  page now (task 20260720-102601).
- `agent-view.ts` `renderAgentPanel`: no longer builds/toggles an inline tool list;
  it sets the model text and the tools link's count (`"N tools"`), hiding the link
  when there are no tools. The link's `href="/settings/"` is static markup, so a
  click is a plain cross-page nav.
- `style.css`: slimmed `.chat__head` (gap + a bottom border to anchor it), added
  `.agent-tools-link` (a subtle pill that brightens on hover/focus), and removed
  the now-dead `.agent-bar` / `.agent-tools*` rules.

## Tests / verification

- `agent-view.test.ts`: `renderAgentPanel` now asserts the model text, the
  `"2 tools"` link label, `href="/settings/"`, and that the link hides at 0 tools.
  Dropped the head's hostile-tool-name test (the head no longer renders tool
  strings; escaping is covered on the settings view). 78 frontend tests green.
- Built `dist/index.html` confirmed: no `card__title` / `agent-tools-toggle` /
  inline `#agent-tools`; the `agent-tools-link` with `href="/settings/"` ships.
  Slim-head visuals are eyeball-verified (per `frontend-verify-needs-e2e-serve`).
