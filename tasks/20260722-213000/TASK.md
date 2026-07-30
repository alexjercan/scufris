# Settings: interactive 'try it' tool runner UI (form + confirm + result)

- STATUS: CLOSED
- PRIORITY: 20
- TAGS: feature,agent,ui,frontend
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

As a homelab operator, I want to click a scufris tool on the Settings page, fill a
form generated from its parameters, confirm, and see the tool's result rendered
inline - WITHOUT a chat turn - so I can debug one tool in isolation. This is the UI
half of the "try it" runner; it consumes the backend endpoint + parameter schema
from task 20260720-134545.

Consent model (GOAL.md): a confirm step gates every Run; the endpoint already
refuses disabled tools; no new setting.

## Steps

- [x] Add a `runTool(name, args)` method to the `SettingsActions` type
      (web/src/settings-view.ts) and wire it in `orchestratorGlobalActions`
      (web/src/agent-settings-view.ts) to `POST /api/agent/tools/{name}/run` with body
      `{args}`, returning the `{ok, text, structured}` result (or throwing on 4xx).
- [x] In the tool card render (settings-view.ts `toolCard`/`toolControlCard`), add a
      "Try it" affordance that reveals a runner form generated from `tool.parameters`:
      one labelled input per param, typed by `param.type` (text / number /
      checkbox), required params marked; all labels escaped. Hide/disable the runner
      for a disabled tool.
- [x] Add a Run button gated by a confirm step (a two-stage Run -> Confirm, or a
      confirm()-style guard consistent with the existing destructive-toggle confirms
      in this file) that only dispatches `runTool` after confirmation. Collect the
      form values into an args object, coercing number/checkbox inputs to their types.
- [x] Render the result inside the card: pretty-printed JSON when `structured` is
      non-empty, else the `text`, all via `escapeHtml` (inside a <pre>); on a 4xx,
      render the error `detail` in an error style. No raw HTML ever reaches innerHTML.
- [x] Tests (web/src/settings-view.test.ts, vitest+jsdom): form is generated from
      `parameters` with correct input types + required; Run does not call the injected
      `runTool` until confirm; a result containing `<script>` is rendered escaped (not
      executed); the error path renders the message. Drive via injected actions (no
      real fetch), matching the file's existing pure-render test style.
- [x] Docs: add a CHANGELOG.md entry (Keep a Changelog) for the tool runner; update
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

## Closing record

What changed:
- `web/src/common.ts`: new `ToolRunResult` interface ({ok, text, structured}).
- `web/src/settings-view.ts`: added `runTool(name, args)` to `SettingsActions`; a
  `toolRunner(tool, actions)` appended to each ENABLED tool card - a "Try it" toggle
  reveals a form generated from `tool.parameters` (input typed by param.type:
  text/number/checkbox, required marked with "*"). Submit is gated by
  `window.confirm` (consistent with the existing destructive-toggle confirms), then
  `collectArgs` coerces values (number/checkbox) - empty optionals are omitted so the
  tool applies its default - and `runAndRender` renders the result: structured JSON
  pretty-printed when present, else text, ALWAYS via `escapeHtml` inside a <pre>; a
  4xx renders the thrown `detail` in an error style.
- `web/src/agent-settings-view.ts`: wired `runTool` in `orchestratorGlobalActions` to
  `POST /api/agent/tools/{name}/run` via `sendJson` (throws the server `detail` on 4xx).
- `web/src/style.css`: `.tool-runner*` styles (matching the existing tool-card /
  settings button tokens).
- CHANGELOG.md: Unreleased -> Added entry for the runner + endpoint.
- Tests: four in settings-view.test.ts (`renders runner form from tool parameters`,
  `run tool requires confirm`, `escapes tool run result`, error path). Updated the two
  `SettingsActions` fakes (settings-view.test.ts, agent-settings-view.test.ts) with a
  `runTool` stub so the shared-interface change compiles.

Decisions / difficulties:
- Result escaping: chose `escapeHtml` + `<pre>` innerHTML (the repo convention) over
  `textContent`; the `escapes tool run result` test pins that a `<script>` in the
  output is inert (`querySelector("script")` is null, innerHTML has `&lt;script&gt;`).
- Confirm is on the Run submit (before any `runTool` call), so a denied confirm makes
  zero network calls - pinned by the first half of `run tool requires confirm`.
- Ran the webpack BUILD, not just vitest, after touching the shared `SettingsActions`
  interface + `AgentTool`/`common.ts` types (LESSONS `type-change-fails-strict-tsc-
  not-vitest`, now x3). Build is green; the two fake updates were the constructors
  that would have broken.
- No stale "read-only" doc surface: the read-only mentions in code refer to the
  settings STORE and the health panels, not the page as a whole; the runner is a
  separate capability and touches neither.

Verification (dev shell, worktree): `npm run format:check` clean, `npm run lint`
clean, `npm run test` 159 passed (4 new), `npm run build` compiled. The three
DoD `test:` proofs pass by name; `manual:` (run host_stats live, see JSON inline)
is PENDING user acceptance - batched to the flow Finish.

Self-reflection: smooth because the backend task had already fixed the type mirror
and the plan named exact anchors. The one deliberate care point was running the
build gate this time (the prior task's retro lesson), applied forward without being
reminded.
