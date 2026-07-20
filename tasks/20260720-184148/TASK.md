# Settings UI: interactive config controls + tools editing

- STATUS: OPEN
- PRIORITY: 32
- TAGS: feature,agent,ui

## Story

As the operator, I want the settings page controls to actually CHANGE things:
toggle the agent on/off, toggle tools, switch model/backend, enable/disable
individual tools, and add/remove MCP servers - each applied via the write
endpoint with a confirm, and reflecting the persisted state.

## Steps

- [ ] In `web/src/settings-view.ts`, when `config.writable` is true, render the
      agent-config card rows as interactive controls: toggles for
      `enabled`/`tools_enabled`, a `model` input, a `backend` select. When
      false, keep the current read-only rows.
- [ ] Wire each control to `PATCH /api/agent/config` via a `fetchJson` helper
      (add a typed patch helper); on success re-render from the returned
      effective config (single authoritative render - do not keep a parallel
      client copy).
- [ ] Add a confirm step before applying a mutation (a small inline confirm or
      `window.confirm`), so a stray click cannot flip the agent off silently.
- [ ] Tools card: render each tool with an enable/disable toggle (from
      `AgentTool.enabled`), wired to the write path. Add an "add MCP server"
      form (id, command, args) and a remove control per configured server.
- [ ] Surface a clear read-only banner when `config.writable` is false
      (controls hidden/disabled).
- [ ] jsdom tests (vitest): `renderSettings` shows controls when writable and
      read-only rows when not; a hostile server id/command is escaped in the
      DOM (no injection); toggling calls the patch path (mock fetch).

## Definition of Done

- With writable config, the page shows working toggles/inputs that call the
  write endpoint and re-render from its response
  (test: `settings_writable_renders_controls`; manual: toggle a setting, see
  it persist after reload).
- With `writable=false`, no mutating controls render and a read-only banner
  shows (test: `settings_readonly_hides_controls`).
- Adding an MCP server from the form persists and appears in the list
  (manual: add a server, reload, still there).
- `npm run ci` passes in `web/` (cmd: `cd web && npm run ci`).

## Notes

- Depends on: 20260720-184136 (write endpoint + `writable` flag) and 20260720-184137 (`AgentTool.enabled`,
  editable `mcp_servers`).
- Lessons: escape untrusted strings in element content
  (`escape-only-host-strings-in-element-content`), prefer one authoritative
  render (`prefer-one-authoritative-render-over-a-parallel-client-counter`),
  run the full `npm run ci` not just vitest (`type-change-fails-strict-tsc`).
- Entry points: `web/src/settings-view.ts` `renderSettings`/`startSettings`,
  `web/src/common.ts` `fetchJson`/`escapeHtml`.
- Stale copy to fix (flagged in T1 review 20260720-184136): `settings-view.ts`
  ~line 127 still renders "Read-only. Everything here is set via environment
  variables; restart to change." Gate that on `config.writable` (now returned
  by `/api/agent/config`). The `webpack.config.js:56` "Read-only agent
  settings" comment is similarly stale.
