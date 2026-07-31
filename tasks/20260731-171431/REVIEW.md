# Review: Split the oversized frontend views under the size cap

- TASK: 20260731-171431
- BRANCH: refactor/split-frontend-views
- BASE: master @ 9ed9ad4

## Round 1

- VERDICT: APPROVE
- REVIEWER: primary, in session. The skill's default is an out-of-context
  reviewer; this session carries a standing instruction not to spawn subagents
  unless the user asks. Recorded as an exception rather than worked around. To
  compensate, every load-bearing claim was re-derived MECHANICALLY or by
  SABOTAGE rather than by re-reading - see "Independent derivation" below. A
  fresh-context round is available on request.

### Independent derivation

**1. The plan's central bet is proven, not asserted.** The task chose flat
siblings with no facade on the argument that "`tsc` is what proves none was
missed". That was tested by sabotage rather than trusted: reverting a single
repoint in `agent-view.ts` (`transcriptReply` back to `./agent-chat-view`) makes
the webpack ts-loader build fail with `ERROR in .../agent-view.ts(30,10)` and
`compiled with 1 error`. The tree was restored and the gate re-run green. A
missed repoint is therefore a build failure, which is what the no-facade
decision rests on.

**2. No import cycle was introduced.** Built the module graph over every
non-test `web/src/*.ts` from its relative `from "./x"` specifiers and walked it
for back-edges: **0 cycles**. This matters more here than in the Python splits,
because `agent-chat-types.ts` exists precisely so the log and turn modules can
name `ChatMsg` without importing the module that imports them.

**3. No export name is defined in two modules.** Extracted every top-level
`export` declaration name across all non-test modules and looked for duplicates:
**none**. The split published generic names (`text`, `line`, `card`, `bar`,
`row`, `section`, `button`, `push`), so a collision would have been an easy way
to import the wrong symbol; there is none. Every non-entry module also has at
least one importer, so the split left no orphan.

**4. The three risky behavioural seams in the chat cut were checked against the
base, not assumed.** This is the one commit that is not a pure move, so each
reshaped seam was compared to `git show master:web/src/agent-chat-view.ts`:

| Seam | Base | Now | Equivalent? |
|-|-|-|-|
| stop button | `if (config.cancelTurn) { send.disabled = false; setStopMode(true); ... }`; `stop()` calls `setComposerEnabled(true)` then `setStopMode(false)` | `setStopMode` absorbs `if (stopping) send.disabled = false`; still called only inside `if (config.cancelTurn)` | Yes. The guard is preserved, and on the false path `setComposerEnabled(true)` already re-enables the button |
| image attach | one `if (config.enableImage)` block wiring attachBtn-click, fileInput-change and input-paste | `createImageAttach` built only when `config.enableImage`, wiring the same three listeners | Yes. Same guard, same three listeners |
| palette keys | `if (paletteOpen()) { ...4 keys, each returning... }` then falling through to Enter-to-send | `if (slash.handleKey(event)) return;` where `handleKey` returns `false` when closed AND for any unhandled key | Yes. Fall-through is preserved for unhandled keys while open |

The `renderPalette` -> `refresh` rename was checked specifically, because a
render-only call becoming a re-match would have broken arrow navigation: the
BASE `renderPalette` already began with
`paletteItems = matchSlashCommands(input.value, slashCommands)`, so `refresh` is
that function verbatim under a new name. No behaviour change.

**5. It is a move where it claims to be.** Re-ran the move proof per commit.
`common.ts` 637 = 637 with zero differences (verbatim). `host-view.ts` and
`stats-view.ts` differ only by `function X` -> `export function X` on the 10 and
18 names that crossed a module boundary, plus two prettier re-wraps in the host
split. The chat commit is deliberately NOT a pure move - see R1.1.

**6. Public surface.** Re-ran the surface diff for all four splits from this
session rather than reading the close-out's numbers: `common.ts` 91 -> 91
(0 dropped, 0 added), `agent-chat-view.ts` 0 dropped / 8 added,
`host-view.ts` 0 dropped / 10 added, `stats-view.ts` 0 dropped / 18 added. Every
added name resolves to a real cross-module consumer. The 20260731-171430 failure
mode - deriving the new surface from what callers import rather than from the
base file's exports - did not recur.

### Checks rerun

| Check | Result |
|-|-|
| `cd web && npm run ci` | prettier clean, eslint clean, 22 test files / 258 tests, webpack compiled - matching the recorded baseline |
| `python scripts/check_file_size.py` | green |
| `git rebase master --exec 'python scripts/check_file_size.py'` | executed and passed at every commit (14/14 with the record commits) |
| `nix build .#scufris-web` | built |
| `nix flake check` | all 5 checks passed |
| `tatr check --ledger LESSONS.md` | exit 0 |
| sabotage: revert one repoint | build fails with 1 ts-loader error; restored |
| import-cycle walk over `web/src` | 0 cycles |
| duplicate export names across modules | none |
| modules with no importer | none beyond the webpack entries |
| `rg -c "^\s+it\b"` on the two allowlisted test files | 43 and 35, identical to `master` |
| `git diff master -- scripts/check_file_size.py` | exactly the 4 owned entries removed; the 8 `tests/`, `scufris/app.py` and both `.test.ts` entries untouched |
| `rg -n "export async function apiFetch" web/src` | 1 hit, `common.ts:110` |

### Findings

**R1.1 MINOR - `tasks/20260731-171431/TASK.md`, close-out "Evidence".** The
move-proof paragraph ends "No logic line moved in any commit", and reports
numbers for only three of the four commits. That over-claims. `30c3026` is a
deliberate FUNCTION cut, not a move: normalized code lines go 849 -> 936, with
53 lines present only in the base, because `paletteOpen` -> `isOpen`,
`closePalette` -> `close`, `renderPalette` -> `refresh`, `runTurn` -> `turn.run`,
and direct `msgs.push(...)`/`msgs[msgs.length - 1]` became the `appendMessage`
and `lastMessage` callbacks. Every one of those is a nameable rename or the
planned deps plumbing - the cut is sound, and section 4 above verifies its three
behavioural seams - but the record must say so rather than fold the commit under
a pure-move claim. State the chat commit's numbers and scope the "no logic line
moved" sentence to the other three.

**R1.2 NIT - `web/src/stats-elements.ts:43`.** `push` is a poor name once it is
a module export: the import site reads `import { push } from "./stats-elements"`
and the call reads `push("cpu", stats.cpu_percent)`, while the same file calls
`Array.prototype.push` three lines above and below it. It was a fine file-local
helper and is a weak public one. Rename to `pushHistory`, which is what the
surrounding comment already describes.

**R1.3 NIT - `web/src/stats-elements.ts:3`.** The header says "Both card modules
read from here", but `stats-view.ts` reads `formatUptime`, `labeledSpark`,
`push` and `severity` from it too - three importers, not two. Say so, or drop
the count.

### Responses

All three fixed on this branch before landing.

- R1.1 fixed: the close-out now reports the chat commit's move-proof numbers and
  names each rename, and scopes the pure-move claim to the other three commits.
- R1.2 fixed: `push` -> `pushHistory` at its definition and every call site.
- R1.3 fixed: the header names all three importers.

### Pending manual items

None from this task. The epic's manual acceptance items (20260731-171411) stay
pending until its remaining children land - 20260731-171432 still owns splitting
`agent-chat-view.test.ts` (1183) and `host-view.test.ts` (997), whose ALLOWLIST
entries this task deliberately left in place.
