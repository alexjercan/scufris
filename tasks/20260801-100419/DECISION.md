# Decision: How Telegram and the web say the same thing about a capability

- DATE: 20260803-120000
- STATUS: ACCEPTED
- TASK: 20260801-100419
- TAGS: telegram, backend, frontend, agents

## Context

`20260729-102148` built `AgentDiagnostics` and the `Capability[T]` envelope
(`supported`, `value`); `20260801-100415` put the legacy `/api/agent/*` family
behind it. Both consumers then threw the third state away at their own
boundary, each naming THIS task in the comment:

| Site | Today |
|---|---|
| `scufris/app.py:2893` | unwraps `Capability[UsageQuota]` to `UsageQuota \| None` "to keep the `SettingsOps` signature; the renderer becomes envelope-aware in 20260801-100419" |
| `web/src/agent-settings-view.ts:520` | unwraps to `.value`; "Telling `unsupported` apart from `nothing to report` in the UI is a later task" |
| `web/src/agent-view.ts:148` | unwraps to `.value` for the sidebar meter |

So the remaining defect is in the WORDING, not in the readers.
`render.py:311` tells a claude operator `no usage data (agent disabled or
non-codex backend)`, and the web renders `quota -` for the same state - both
of which read as "something is broken" rather than "this backend has no
account quota".

Only `codex.py:135` reads a quota at all; `claude.py:492`, `opencode.py:262`
and `mock.py:63` all return `Capability.unsupported()`.

`scufris/telegram/` itself is already clean - `rg -n
"resolve_codex_home|read_usage" scufris/telegram/` is empty on base - so the
original Steps 2 and 3 of this task describe work `20260801-100415` already
did.

## Decision

**D1 - one three-state vocabulary, duplicated per language.** Every surface
renders the envelope with the same three readings:

| Envelope | Reading |
|---|---|
| `supported`, value | the measurement |
| `supported`, no value | `nothing reported yet` |
| `unsupported` | `not reported by the <backend> backend` |

The strings cannot cross the Python/TypeScript boundary, so they are
duplicated: constants in `scufris/telegram/text.py`, a `capabilityText` helper
in `web/src/agent-settings-view.ts`. The contract section of
`scufris/README.md` becomes the ONE place that states the vocabulary and the
rule that a new surface consumes `AgentDiagnostics` and renders all three
states.

**D2 - `SettingsOps.usage` is deleted; the quota rides on `OrchestratorInfo`.**
`AccountInfo` is already the service's single answer for "the account behind
this agent" (auth mode, model, enabled, quota). `SettingsOps` asks for those
facts twice today: `info()` rebuilds auth mode/model/enabled by hand and
`usage()` reads the quota separately, so both `/settings` and `/settings usage`
need two calls. `OrchestratorInfo` gains `quota: Capability[UsageQuota]`,
`info()` is built from `diagnostics.account(orchestrator)` plus the two facts
`AccountInfo` does not carry (backend name, permission mode), and `SettingsOps`
drops from five providers to four. The whole of `info()` moves under
`asyncio.to_thread` - the codex reader rglobs and parses every rollout, which
R1.1 of `20260801-100415` already required to stay off the bot's poll loop.

**D3 - `/settings tools` keeps the console tool catalog.** `scufris/README.md`
records that `/api/agent/tools` and `/api/agent/mcp` deliberately do NOT
delegate to the service: they describe the operator console's OWN in-process
tool runner, which does not go through the orchestrator's backend, and the web
orchestrator settings page reads `/api/agent/mcp` for that reason. Telegram's
`tools` provider already reads the same in-process catalog, so it already
agrees with the web. Unchanged - and the reason moves into the contract section
so the next reader does not "fix" it.

**D4 - the landing sidebar usage meter stays a meter.** It is a bar with no
text row; an unsupported backend hides it, which is the honest reading for a
meter. The unwrap stays; only the stale comment is corrected.

## Alternatives considered

- **A server-rendered string inside `Capability`.** Rejected: it puts
  presentation into a wire model the API also serves to non-UI clients, and
  fixes English into the transport.
- **Route Telegram's `tools` through `diagnostics.tools()`.** Rejected: it
  returns `unsupported` for an opencode or mock orchestrator while the web
  still lists the console catalog from `/api/agent/mcp`. That CREATES the
  cross-surface divergence this task exists to remove, and it would drag the
  console tool runner's own semantics into scope.
- **Keep `SettingsOps.usage` and make only the renderer envelope-aware.** The
  minimum change, and rejected only because it leaves two providers answering
  from the same `AccountInfo` - the duplicated `auth_mode_for_backend` call in
  `info()` is exactly the kind of per-call-site backend reasoning the
  diagnostics service was extracted to delete.
- **Land Telegram now and the web separately.** Rejected: the two unwraps are
  one defect and the DoD's manual check spans both surfaces, so splitting means
  shipping an interval where Telegram and the web disagree - the state this
  task is closing.

## Consequences

- `OrchestratorInfo` stops being a bag of plain strings: it carries one pydantic
  envelope. The "no `app` import" property that lets the transport be tested
  standalone is preserved (`Capability` lives in `backends/base.py`).
- `/settings` pays the quota read it already paid (the summary renders the
  primary window); `/settings health` and `/settings tools` never call `info()`,
  so nothing new becomes expensive.
- Two hand-maintained copies of the D1 strings. The contract section is the
  guard: it names the vocabulary, and the DoD greps for the old wording.
- `AgentSettingsData.usage`/`.memory` change type, so
  `web/src/agent-settings-view.test.ts` fixtures move to the envelope shape.
- Not fixed here: `scufris/health.py:258` still counts CODEX sessions for a
  claude or opencode orchestrator on both health surfaces. Owned by
  `20260803-032950`; `/settings health` inherits that fix when it lands.
