# Retro: read-only settings/config page

- DATE: 20260720
- VERDICT: shipped

## What went well

- The multipage machinery paid off exactly as the ledger promised: adding a third
  page was a mechanical repeat of the Agent/Stats pattern (one entry + one
  HtmlWebpackPlugin + one historyApiFallback rewrite + a nav link), and
  `StaticFiles(html=True)` served `/settings/` with zero backend routing changes.
  No surprises, per `webpack-multipage-htmlplugin-per-page`.
- Resolving the page-vs-panel open question up front (the user said "settings
  page", and it is the future home for editable settings) kept the build focused.
- The e2e serve+curl caught nothing broken but confirmed the whole slice wires,
  which is the point of that step for a brand-new page/endpoint.

## What went wrong / friction

- First e2e attempt returned nothing because `python -m scufris` needs the `serve`
  subcommand (it is a CLI, not a bare server). A quick fix once the server log
  showed it never bound.
- Self-review found a genuine UX contradiction I had shipped into the first commit:
  the Tools section listed the tool catalog even when `agent_tools_enabled=False`,
  because `/api/agent/tools` enumerates the MCP tools regardless of the flag. On a
  page whose whole job is to explain the setup, "tools: disabled" next to six tool
  cards is exactly the confusion it should remove. Fixed to say "tools are disabled"
  with no cards.

## Lesson

- `an-info-view-must-not-contradict-itself` - when a read-only status page pulls
  from two sources (here a config flag and a catalog endpoint that ignores it),
  reconcile them in the view: if the flag says a capability is off, do not render
  the capability's catalog as if it were live. The page exists to remove confusion,
  so an internal contradiction is a real bug, not cosmetic. (No standalone ledger
  entry - captured here; watch for recurrence before promoting.)

## Follow-ups

- Unblocks 20260720-102600 (chat head redesign): the head's `tools` toggle can now
  be removed and pointed at this page.
- Editable settings / switching the model is still deferred to a later spike
  (writing config back + restarting the agent cleanly).
