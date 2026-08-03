# Review: Align Telegram and the UI with orchestrator diagnostics

- TASK: 20260801-100419
- BRANCH: fix/telegram-ui-diagnostics-alignment

## Round 1

- REVIEWER: out-of-context
- VERDICT: APPROVE

No BLOCKER or MAJOR. The Story is delivered: both surfaces read the three
`Capability` states, `SettingsOps.usage` is gone, and the two Telegram tests
drive the real app per backend against the real `AgentDiagnostics` rather than
a hand-written table. All nine Steps deliver their literal text.

- [ ] R1.1 (MINOR) scufris/README.md:356 - the `Consuming surfaces` table this
  diff adds is the one source the two language-local copies point back to, and
  it names `capabilityText` in `web/src/agent-settings-view.ts`; the same diff
  moved that helper to `web/src/agent-settings-panels.ts:140`. Same stale
  pointer at `scufris/telegram/text.py:58` and `web/src/agent-settings-view.ts:62`.
  Repoint all three at `web/src/agent-settings-panels.ts`.
  - Response:

- [ ] R1.2 (MINOR) scufris/telegram/render.py:388 - `render_settings_summary`
  reads only `usage.primary`, but `primary` and `secondary` are independently
  optional (`scufris/sessions/models.py:26-27`) and `scufris/sessions/usage.py:48`
  builds both from separate rollout keys. A quota with `primary=None` and a
  `secondary` window prints `usage: nothing reported yet` in the summary while
  `/settings usage` prints the secondary window - the disagreement
  `test_render_settings_summary_carries_the_capability_reading` says cannot
  happen. Fall back to `usage.secondary` when `primary is None`, and add that
  case to the test.
  - Response:

- [ ] R1.3 (MINOR) CHANGELOG.md:8 - no entry, though this changes what operators
  read on `/settings`, `/settings usage` and the agent settings page, and
  AGENTS.md:83 requires a `CHANGELOG.md` update for a notable change (both
  sibling tasks in this epic added their own bullet). Add a bullet under
  `## [Unreleased]` / `### Changed` for the three-state wording
  (`nothing reported yet` / `not reported by the <backend> backend`) replacing
  `no usage data (agent disabled or non-codex backend)` and the bare `-`.
  - Response:

- [ ] R1.4 (MINOR) scufris/telegram/render.py:325 - the `windows` comprehension
  plus the `usage is not None` guard at line 340 spend 12 lines where 3 did the
  same work. Replace lines 325-336 with
  `if usage is None or (usage.primary is None and usage.secondary is None): return _fenced("Usage", _quota_reading(info))`,
  restore the
  `for label, window in (("primary", usage.primary), ("secondary", usage.secondary)):`
  loop with a `if window is None: continue`, and drop the `usage is not None and`
  at line 340 - the early return narrows `usage` for mypy.
  - Response:

- [ ] R1.5 (MINOR) web/src/agent-view.ts:151 - the new comment ends
  `(DECISION D4 of tasks/20260801-100419)`. AGENTS.md:103: task IDs belong in
  task records and Markdown, never in code comments. Drop the parenthetical; the
  two preceding sentences already state the invariant as a fact about the code.
  The Step's literal text asked for the citation, so Step and repo rule conflict
  here - the repo rule wins.
  - Response:

- [ ] R1.6 (NIT) web/src/agent-settings-panels.ts:47 - `capabilityPanel`,
  `resetsIn` (line 116) and `capabilityText` (line 140) are exported but used
  only inside this module; `agent-settings-view.ts:34` imports neither. Drop
  `export` from those three.
  - Response:

Verified in the recording pass, independently of the out-of-context reviewer:

- `ruff check . && mypy . && python -m pytest`: clean, 989 passed 1 skipped;
  `cd web && npm run ci`: 261 passed, webpack build clean. Both match the
  close-out's numbers.
- The single skip is the `CODEX` case of
  `test_telegram_hides_codex_account_data_for_other_backends` - the one backend
  that does read a quota, so the parametrized carve-out is legitimate.
- All four `cmd:` DoD greps green, including the two red on base.
- R1.1 re-derived by grep: `capabilityText` is defined only at
  `agent-settings-panels.ts:140`, and three prose references still name the old
  file.
- R1.2 re-derived by reading `usage.py:36-50` and `models.py:26-27`: the two
  windows come from separate rollout keys and either can be absent alone.

Pending user check: the `manual:` DoD item - backend and account information
feels consistent across the landing page, agent settings and Telegram.

- Process signal: the `agent-settings-panels.ts` split was forced by the
  600-line ratchet, not chosen; recorded in the close-out but not DECISION.md.
  Acceptable as a mechanical move, worth a retro note on planning around the
  ratchet.
- Observation: `resetsIn` is duplicated verbatim in `web/src/chat-sidebar.ts:96`
  and `agent-settings-panels.ts:116`. Pre-existing; candidate follow-up dedupe
  into `common.ts`.
