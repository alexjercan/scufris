# Settings: interactive 'try it' tool runner UI (form + confirm + result)

- STATUS: OPEN
- PRIORITY: 20
- TAGS: feature,agent,ui,frontend

## Story

As a homelab operator, I want to click a scufris tool on the Settings page, fill a
form generated from its parameters, confirm, and see the tool's result rendered
inline - WITHOUT a chat turn - so I can debug one tool in isolation. This is the UI
half of the "try it" runner; it consumes the backend endpoint + parameter schema
from task 20260720-134545.

Consent model (GOAL.md): a confirm step gates every Run; the endpoint already
refuses disabled tools; no new setting.

## Steps

- [ ] Add a `runTool(name, args)` method to the `SettingsActions` type
      (web/src/settings-view.ts) and wire it in `orchestratorGlobalActions`
      (web/src/agent-settings-view.ts) to `POST /api/agent/tools/{name}/run` with body
      `{args}`, returning the `{ok, text, structured}` result (or throwing on 4xx).
- [ ] In the tool card render (settings-view.ts `toolCard`/`toolControlCard`), add a
      "Try it" affordance that reveals a runner form generated from `tool.parameters`:
      one labelled input per param, typed by `param.type` (text / number /
      checkbox), required params marked; all labels escaped. Hide/disable the runner
      for a disabled tool.
- [ ] Add a Run button gated by a confirm step (a two-stage Run -> Confirm, or a
      confirm()-style guard consistent with the existing destructive-toggle confirms
      in this file) that only dispatches `runTool` after confirmation. Collect the
      form values into an args object, coercing number/checkbox inputs to their types.
- [ ] Render the result inside the card: pretty-printed JSON when `structured` is
      non-empty, else the `text`, all via `escapeHtml` (inside a <pre>); on a 4xx,
      render the error `detail` in an error style. No raw HTML ever reaches innerHTML.
- [ ] Tests (web/src/settings-view.test.ts, vitest+jsdom): form is generated from
      `parameters` with correct input types + required; Run does not call the injected
      `runTool` until confirm; a result containing `<script>` is rendered escaped (not
      executed); the error path renders the message. Drive via injected actions (no
      real fetch), matching the file's existing pure-render test style.
- [ ] Docs: add a CHANGELOG.md entry (Keep a Changelog) for the tool runner; update
      any Settings/README surface that describes the page as read-only.

## Definition of Done

- Clicking a tool's "Try it" renders a typed form from its `parameters`
  (test: `renders runner form from tool parameters`).
- Run is gated behind a confirm step and calls the run action only after confirm
  (test: `run tool requires confirm`).
- The result and any error render escaped - a `<script>` in the result is inert
  (test: `escapes tool run result`).
- manual: on the running app, open Settings, pick host_stats, Run, and see the JSON
  result inline with no chat turn.
- Frontend gate green (cmd: `npm run test` in web/) and it builds
  (cmd: `npm run build` in web/).

## Notes

- Relevant files: web/src/settings-view.ts (`toolCard` ~56, `toolControlCard` ~225,
  `renderToolControls` ~243, `SettingsActions`), web/src/agent-settings-view.ts
  (`orchestratorGlobalActions` ~377, tools render ~341), web/src/common.ts
  (`AgentTool` ~190 - now carries `parameters` after task 20260720-134545),
  `escapeHtml`/`el`/`fetchJson` in web/src/common.ts.
- The render helpers are pure and side-effect-free (jsdom drives them fetch-free via
  injected `SettingsActions`); keep the runner the same shape so tests need no fetch.
- Depends on: 20260720-134545 (backend run endpoint + `parameters` contract).
