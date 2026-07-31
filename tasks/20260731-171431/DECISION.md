# Decision: flat sibling modules, no facade, and `common.ts` splits by concern

- STATUS: ACCEPTED
- DATE: 2026-07-31
- TASK: 20260731-171431
- TAGS: refactor, maintainability, kiss, frontend
- EPIC: 20260731-171411

## Context

Four modules under `web/src/` are over the 600-line source cap and allowlisted
in `scripts/check_file_size.py`. Confirmed on `8bfbe74`, not trusted from the
epic record:

| Module | Lines | Over by |
|-|-|-|
| `web/src/agent-chat-view.ts` | 1106 | 506 |
| `web/src/host-view.ts` | 1022 | 422 |
| `web/src/stats-view.ts` | 870 | 270 |
| `web/src/common.ts` | 834 | 234 |

Baseline before any edit: `python scripts/check_file_size.py` exits 0,
`cd web && npm run ci` is green (22 test files, 258 tests, webpack build
succeeds).

Three sibling tasks settled the epic's shape for PYTHON: an oversized module
becomes a package of the same name with a facade `__init__.py`, so import paths
do not move and no call site changes. This is the first frontend child, and the
question the epic left open is whether that shape translates.

## Decision

### 1. No facade, no barrel, no directory - flat siblings, importers repointed

The Python facade exists to answer a problem TypeScript does not have. In
Python the import path is a runtime string: 61-plus call sites, `monkeypatch`
targets written as `"scufris.mcp_host_tools._inspector"`, and `dir(module)`
enumerations all address a module by name, and a moved name fails SILENTLY -
20260731-171428 shipped a patch that patched nothing, and 20260731-171430 had to
diff `ast` against `dir()` to find eight names a facade had quietly dropped.

None of that holds here, measured:

- Every importer of these four modules is a static `import ... from "./x"`
  checked by `tsc`. A missed repoint is a build failure in `npm run ci`, not a
  silent no-op.
- `rg 'vi\.mock|vi\.spyOn' web/src` returns nine hits and every one targets a
  DOM global (`window.confirm`, `HTMLAnchorElement.prototype.click`). There is
  no string-addressed module target anywhere in the frontend tests, so the
  silent-failure mode the facade was protecting against does not exist here.
- `web/src` is 53 flat modules with zero `index.ts` and zero re-export-only
  module (`rg '^export \*|export \{.*\} from'` finds nothing). `chat-stream.ts`,
  `chat-format.ts`, `markdown.ts` and `chat-commands.ts` are the existing
  precedent: `agent-chat-view.ts` imports them directly as siblings.
- webpack pins only the nine per-page ENTRY files
  (`agent.ts`, `stats.ts`, `settings.ts`, `projects.ts`, `agents.ts`, `host.ts`,
  `agent-detail.ts`, `project-detail.ts`, `login.ts`). None of the four files
  here is an entry; `webpack-partials.js` references no `src/*.ts` at all.

So the frontend answer is the opposite of the Python one, for the reason the
Python one existed: import-path stability buys nothing when the compiler
enforces it, and a barrel would re-create the god-import this epic exists to
break. Every new module is a sibling `web/src/<name>.ts`, and each importer
names the module it actually needs.

### 2. `common.ts` splits BY CONCERN into three type modules; it does not become
a directory

The Fog question, settled by measurement. `common.ts` is 834 lines of which
roughly 650 are pure `interface`/`type` declarations, so the obvious cut is
"types out, runtime stays". That cut does not fit: the type region measures
about 653 lines against a 600 cap, so a single `types.ts` fails the guard on
arrival. The types have to split, and the seam is not invented - it is the one
the consumers already draw:

- `stats-view.ts` imports `Availability`, `DiskIoRate`, `GpuStats`,
  `HostOverview`, `HostStats`, `UnitList` - every one from `common.ts:3-211`.
- `host-view.ts` imports `HostActionRecord`, `HostAuditRecord`,
  `HostConfirmation`, `HostDigest`, `HostDigestView`, `HostPreview`, `HostStep`,
  `ScheduleState` - every one from `common.ts:212-333`.

Zero overlap. The two halves have disjoint consumers, so they are two modules:

| Module | Owns | ~Lines |
|-|-|-|
| `stats-types.ts` | the `/api/stats` and `/api/host/overview` wire shapes: `MemStats` .. `HostOverview`, minus `AppConfig` | 200 |
| `host-types.ts` | the `/api/host/actions` and `/api/host/audit` wire shapes: `HostStep` .. `HostDigestView` | 122 |
| `agent-types.ts` | the chat, agent, session, usage and project wire shapes: `ToolCall` .. `BackendOption` | 320 |
| `common.ts` | the runtime: `AppConfig` + `loadConfig`, `escapeHtml`, `el`, `formatBytes`, the CSRF constants, `csrfToken`, `goToLogin`, `apiFetch`, `logout`, `fetchJson`, `sendJson`, and the shared display labels (`ORCHESTRATOR_ID`, `PERMISSION_MODES`, `authLabel`, `BACKEND_LABELS`, `backendLabel`, the two poll defaults) | 200 |

`apiFetch` and the auth bootstrap contract STAY IN `common.ts`, in one place,
unduplicated - which is also what `AGENTS.md` pins ("Authenticated frontend API
calls: `apiFetch` in `web/src/common.ts`") and what `web/README.md` documents.
Nothing in the CSRF/401 seam moves at all. `AppConfig` stays beside its only
reader, `loadConfig`.

Not a directory: `web/src/common/` would need either an `index.ts` (a barrel,
rejected above) or `./common/common` at every call site. Flat siblings are the
repo's measured idiom.

### 3. The three views split into siblings named for the seam

**`agent-chat-view.ts`** (1106). 710 of those lines are ONE function,
`createAgentChat`, so 20260731-171429's rule binds: measure the groups inside it
before choosing the cut. Measured, within lines 304-1013:

| Group | Lines |
|-|-|
| DOM skeleton + scroll/pill/render helpers | 80 + 98 |
| image attach (`renderAttachPreview`, `acceptImage`, its wiring) | 65 |
| streaming turn (`runTurn`) | 216 |
| slash palette (`renderPalette`, `runCommand`, `closePalette`, its key handling) | 90 |
| edit-to-fork, composer wiring, control handle, mount | 161 |

Extracting only the module-level helpers leaves the function at 710, over the
cap on its own, so the function is cut - twice, at the two seams that own their
own state:

| Module | Owns | ~Lines |
|-|-|-|
| `agent-chat-types.ts` | `ChatMsg`, `AgentChatConfig`, `ChatControl`, `RenderChatOpts` | 105 |
| `agent-chat-log.ts` | the pure log rendering: `distinctTools`, `messageMeta`, `transcriptReply`, `copyButton`, `messageFoot`, `renderChatLog` | 185 |
| `agent-chat-turn.ts` | `createTurnRunner`: the pending bubble, the throttled markdown paint, the thinking spoiler, settle/cancel/fail, and the `streaming` + `cancelCurrent` state | 245 |
| `agent-chat-composer.ts` | `createSlashPalette` (owns `paletteItems`/`paletteIdx`) and `createImageAttach` (owns `pendingImage`), plus `autosize` | 175 |
| `agent-chat-view.ts` | `createAgentChat` (DOM build, render, edit-to-fork, wiring, the control handle) and `startAgentChat` | 480 |

`createTurnRunner` is a state-owning collaborator, not a size trick: `streaming`
and `cancelCurrent` are the turn's own state, read by `submit`, `forkFrom`,
`requestCancel` and the mount-time reattach guard. It takes explicit deps
(`log`, `config`, `appendMessage`, `lastMessage`, `render`, `maybeScroll`,
`setComposerEnabled`, `setStopMode`, `onSettled`) rather than the component
itself - passing the component would be the mixin coupling 171429 rejected.
`appendMessage`/`lastMessage` are callbacks rather than the `msgs` array because
`createAgentChat` REASSIGNS `msgs` (`setMessages`, `reset`, `forkFrom`), so a
captured array reference would go stale.

`agent-chat-types.ts` exists so `agent-chat-log.ts` and `agent-chat-turn.ts` can
name `ChatMsg` without importing from the module that imports them.

**`host-view.ts`** (1022):

| Module | Owns | ~Lines |
|-|-|-|
| `host-format.ts` | the text-only building blocks and the read-only formatters: `text`, `line`, `button`, `section`, `RISK_WORD`, `riskBadge`, `formatExpiry`, `expiryMillis`, `formatAgo`, `formatRequester`, `staleReason` | 140 |
| `host-actions.ts` | the `HostActions` contract, `dispatch`, and the last-error state it writes (`hostError`, `_resetHostError`) | 100 |
| `host-proposal.ts` | the pending queue: `commands`, `preview`, `undoLine`, `denyControls`, `approveControls`, `pendingCard` | 210 |
| `host-history.ts` | what already happened: `resultRows`, `decidedCard`, `auditTable` | 215 |
| `host-checks.ts` | the scheduled checks: `scheduleRow`, `digestCard`, `checksSection` | 130 |
| `host-view.ts` | `HostViewData`, `isTyping`, `renderHost`, the three readers, `startHost`, `POLL_SECONDS` | 320 |

`dispatch` moves with the `lastError` module state it writes, because they are
one mechanism: every mutating control funnels through `dispatch`, and
`renderHost` reads `hostError()`. Splitting them would put a write in one module
and its only read in another.

The two rules in the file's header comment survive the cut and are restated
where they now apply: `host-format.ts` becomes the only place text is set (its
`text()`/`line()` are the sole sinks, `el()` is never called with its html
argument), and `host-proposal.ts` is the single module holding the
one-way-approve control, so "a one-way action has no ordinary approve button"
stays reviewable in one file.

**`stats-view.ts`** (870):

| Module | Owns | ~Lines |
|-|-|-|
| `stats-elements.ts` | the pieces BOTH card modules read: `formatUptime`, `severity`, `tempSeverity`, `bar`, `card`, and the rolling history + `sparkline`/`labeledSpark`/`_resetStatsHistory` | 145 |
| `stats-cards.ts` | the live gauges: cpu/memory/load/disks/network/gpu cards and their helpers (`baseDisks`, `diskTempReadings`, `perSec`, `row`, `rate`, the coretemp readers) | 315 |
| `stats-host-cards.ts` | the `/api/host/overview` cards: `hostCard`/`hostRow`/`hostValue`, `availabilityNote`, `noteRow`, `failedUnitsCard`, `generationsCard`, `nixStoreCard`, `thermalCard` | 270 |
| `stats-view.ts` | `renderSummary`, `renderCards`, `renderHostCards`, `totalDiskIo`, `totalNetIo`, `setStatus`, `refresh`, `refreshHost`, `markHostCardsStale`, `startStats` | 205 |

`stats-elements.ts` is one module rather than two 50-line ones because both card
modules read from it and a `stats-format` / `stats-spark` pair would only add a
file. It exists because `stats-cards.ts` and `stats-host-cards.ts` both need
`severity`/`bar`/`card`/`tempSeverity`; without it they would import each other.

The `textContent`-never-`innerHTML` rule that the host cards carry (the fix for
the stored XSS a `<img src=x onerror=...>.service` unit name produced) moves
into `stats-host-cards.ts` with the helpers it governs, stated as the invariant
rather than as a pointer to the ledger key.

### 4. `web/src/agent-chat-view.test.ts` and `web/src/host-view.test.ts` belong
to 20260731-171432; NO test code moves in this task

Both files are over the 900-line test cap and both are allowlisted.
20260731-171432's Steps name them explicitly. This task therefore:

- does NOT delete either allowlist entry, and does not create any `.test.ts`
  file;
- changes only IMPORT LINES in the test files, repointing names to the sibling
  they moved to (`renderChatLog` -> `./agent-chat-log`, `formatExpiry` ->
  `./host-format`, and so on);
- moves no assertion, no fixture and no `describe` block between files.

That is not just a scope courtesy, it is what the guard requires: moving test
code out of `agent-chat-view.test.ts` (1183) or `host-view.test.ts` (997) could
drop either below 900, and an entry whose file is back inside the cap fails as
STALE. This task cannot partially split those files without breaking the gate it
is here to satisfy.

The four entries this task deletes are exactly
`web/src/agent-chat-view.ts`, `web/src/common.ts`, `web/src/host-view.ts` and
`web/src/stats-view.ts`. The remaining entries stay: `scufris/app.py` and
`tests/test_app.py` (20260729-103712), the six other `tests/*.py` plus the two
web `.test.ts` files (20260731-171432).

### 5. Four commits, one per over-cap file, in dependency order

`common.ts` first, because the other three import from it and repointing their
type imports is part of that commit. Then one commit per view, each deleting its
own `ALLOWLIST` entry, so the guard is green at EVERY commit rather than only at
the tip - proved with
`git rebase master --exec 'python scripts/check_file_size.py'`, not asserted.
This is the 20260731-171430 shape (independent files) rather than
20260731-171429's single indivisible commit: nothing here becomes a package, so
no intermediate state is unrepresentable.

## Consequences

- Roughly 31 files change one or two import lines (the `import type { ... } from
  "./common"` statement splits by domain). `tsc` in `npm run ci` is the proof
  that none was missed.
- `agent-settings-view.ts` is 589 lines, 11 under the cap. Splitting its
  `import type` statement adds lines. If it crosses 600 the guard fails
  immediately and loudly; the plan checks every touched file's count before
  committing rather than after.
- `web/README.md`'s "View logic is separated from the DOM entry" convention
  describes a two-file page (`<page>.ts` + `<page>-view.ts`) and must be widened
  to the seam-named siblings.
- Task records and LESSONS.md entries citing `common.ts:<line>`,
  `host-view.ts:<line>`, `stats-view.ts:<line>` or `agent-chat-view.ts:<line>`
  no longer resolve. They are history and are not rewritten.
- Comment sweep, per the `AGENTS.md` policy: the review-round citations in
  `host-view.ts` ("review round 1, R1.1" at 152, "R1.2" at 121, "R1.3" at 843,
  "R1.4" at 192) and the ledger-key citations in `host-view.ts:135,898` and
  `stats-view.ts:54,563` lose the lore and keep the invariant as a fact about
  the code. Confirm the exact set with a grep at work time rather than from this
  list.

## Alternatives considered

- **Keep `common.ts` as a facade re-exporting the type modules** (`export type *
  from "./stats-types"`). Zero importer churn, and it is the Python answer.
  Rejected: it preserves the single god-import the epic is trying to break,
  `tsc` already makes the churn safe and mechanical, and the repo has no
  re-export module to be consistent with.
- **One `types.ts` for all of `common.ts`'s types.** Rejected on the
  measurement: about 653 lines against a 600 cap, so it fails the guard on
  arrival.
- **`web/src/common/` as a directory.** Rejected: it needs a barrel or ugly
  `./common/common` paths, and `web/src` has no nested directory today.
- **Split `agent-chat-view.ts` only at its module-level helpers.** Rejected on
  the measurement: `createAgentChat` alone is 710 lines, so the function must be
  cut regardless.
- **Cut `createAgentChat` at the DOM-construction seam instead of the turn
  seam.** Viable on size but the DOM build is 80 lines of straight-line
  construction with no state; the turn runner is 216 lines that own `streaming`
  and `cancelCurrent`. The seam with state is the one worth a module.
- **Split `host-view.ts` into a card module per section and leave `dispatch` in
  the view.** Rejected: every control calls `dispatch`, so it would be imported
  back from the module that imports the cards.
- **Move the two web `.test.ts` files' allowlist entries here.** Rejected in
  section 4: it would either break 20260731-171432's scope or fail the guard on
  a stale entry.
