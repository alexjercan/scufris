# Settings UI: interactive config controls + tools editing

- STATUS: CLOSED
- PRIORITY: 32
- TAGS: feature,agent,ui

## Story

As the operator, I want the settings page controls to actually CHANGE things:
toggle the agent on/off, toggle tools, switch model/backend, enable/disable
individual tools, and add/remove MCP servers - each applied via the write
endpoint with a confirm, and reflecting the persisted state.

## Steps

- [x] Interactive Agent card when `config.writable`: `enabled`/`tools`
      toggles, `model` text input, `backend` select (`renderAgentControls` +
      `toggleRow`/`selectRow`/`textRow`). Read-only rows otherwise.
- [x] Controls dispatch through a `SettingsActions` seam wired in
      `startSettings` to `sendJson` (new common.ts helper carrying the server's
      `detail` on error); after any mutation the page RELOADS from the server
      and re-renders (single authoritative render, no parallel client copy).
- [x] Confirm before a high-impact turn-OFF (agent enabled, all tools) and
      before removing a server; a cancelled confirm reverts the toggle.
- [x] Tools card: per-tool enable/disable toggle (from `AgentTool.enabled`) that
      rebuilds the full `disabled_tools` set; "add MCP server" form (id,
      command, args) via `POST /api/agent/mcp_servers` and a remove button per
      CONFIGURED server via `DELETE .../{id}` (built-in scufris has none).
- [x] Read-only banner when `config.writable` is false (controls hidden).
- [x] jsdom tests: controls-when-writable, read-only-banner-when-not,
      confirmed/cancelled disable, tool-toggle sends full set, add-server form,
      remove-configured-only, hostile-id escaped.
- [x] NOTE: added two BACKEND endpoints this task needed and could not be split
      out without shim code - `POST /api/agent/mcp_servers` and `DELETE
      /api/agent/mcp_servers/{id}` (incremental, since the config response
      exposes only id+source, so the client cannot rebuild the whole spec list).

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
  settings" comment is similarly stale. DONE: the stale copy is gone (writable
  view has no "restart to change"; a jsdom test asserts its absence), the
  file-header comment now says "interactive when writable", and the webpack
  comment was refreshed.

## Close-out

- The writable/read-only split is driven by `live = config.writable &&
  actions !== null`. `renderSettings` stays pure: interactivity comes from an
  injected `SettingsActions` seam (patch/addServer/removeServer/reload), so
  jsdom tests drive controls with fakes and `startSettings` wires the real
  `sendJson` calls + a `reload()` that re-fetches and re-renders. This keeps
  the single-authoritative-render property (no client-side config copy).
- Needed backend work mid-task: making MCP add/remove correct required
  incremental endpoints because `GET /api/agent/config` only exposes
  `{id, source}` per server, so the client cannot resend the full spec list
  for a whole-list PATCH. Added `POST /api/agent/mcp_servers` and
  `DELETE /api/agent/mcp_servers/{id}` (the server rebuilds the list from
  `settings.mcp_servers`). This is the flow "inseparable slice" case - a
  frontend-only task would have needed throwaway shim code. Extracted the id
  validation into `_validate_mcp_spec` shared by PATCH and POST.
- `type-change-fails-strict-tsc` bit as predicted: adding required `enabled`
  to `AgentTool` and `writable` to `AgentConfig` passed `vitest` but the
  webpack build failed on `agent-view.test.ts`'s tool factory. Ran the full
  `npm run ci` (the real type gate), not just vitest.
- `require-await`/`no-unnecessary-type-assertion` eslint: await-less async fake
  methods must be plain `() => Promise.resolve()`, and `querySelectorAll("input")`
  already yields `HTMLInputElement` so the casts were redundant.
- Verified end to end by serving the built bundle through uvicorn and curling:
  `/settings/` 200, config `writable=true`, POST add -> [scufris, fs], PATCH
  model -> gpt-5.6, DELETE -> [scufris], bad id -> 422. Cleaned the state file
  the e2e run wrote to `~/.local/state/scufris`.
- Self-reflection: should have grepped every `AgentTool`/`AgentConfig` literal
  across web/src BEFORE the first `npm run ci` (the shared-type-change lesson
  literally says to) - would have caught the agent-view.test factory in one
  pass instead of via a build failure.
