# Review: agent page - context breakdown + weekly-usage panel

- DATE: 20260719
- VERDICT: APPROVE (1 round)

## Scope reviewed

Frontend: `common.ts` (`SessionContext`/`RateWindow`/`UsageQuota` types),
`index.html` (`.sidebar__foot` with `#context-panel` + `#usage-meter`),
`agent-view.ts` (`renderContext`/`renderUsage` + `loadContext`/`loadUsage` +
`refreshSidebar`), `style.css`, `agent-view.test.ts` (4 new tests). Backend: a
correctness fix to `read_context` (+ test).

## Correctness

- **Data-correctness fix found in review.** The context bar uses
  `input_tokens / context_window`, but `read_context` filled `input_tokens` from
  `total_token_usage` (cumulative across turns). Real data confirmed the bug: a
  2-turn session had `total_in=58458` vs the true current fill `last_in=15263`
  (of 258400) - the bar would have read ~23% instead of ~6%. Fixed to prefer
  `last_token_usage` for the occupancy fields (input/cached), keeping
  output/reasoning/total cumulative; falls back to `total` when a session
  predates the field (so the older tests and real 1-turn sessions still hold).
  Pinned by a new test and re-verified live: the same 2-turn session now reads
  `fill 14497 -> 5.6%`.
- `renderContext`/`renderUsage` hide when their data is null (no active session /
  no reported limit). The `.usage-block` sets `display:flex`, which beats the UA
  `[hidden]` rule, so an explicit `.usage-block[hidden] { display:none }` restores
  hiding - easy to miss, covered by the "hides when..." tests.
- Escaping: `usageRow` escapes both label and value, so `plan_type` (from codex)
  is safe; numbers go through `toFixed`. No host strings rendered raw.
- The weekly window is labelled from `window_minutes >= 10080`; `resetsIn`
  formats the epoch to "Xd Yh"/"Xh Ym"/"Xm" and "-" when null. Secondary window
  rendered when present.
- Sidebar became a bounded flex column (`max-height: calc(100vh - 2rem)`), so the
  session list scrolls (`flex:1`) while the two stat blocks stay pinned at the
  foot; mobile drops the sticky/height so it stacks.
- `refreshSidebar` (list + context + usage) runs on start, after each reply, and
  on switch/new, so both panels track the current session and latest turn.
- Full suites green: frontend `npm run ci` (37 jsdom tests + build), backend
  `ruff`/`ruff format`/`mypy`/`pytest`. Live: `/api/agent/usage` -> real weekly
  window (`plus / 10080 / 1.0%`); bundle ships `renderContext`/`renderUsage`.

## Nits (non-blocking)

- Usage/context refresh only when a turn runs (codex emits `token_count`
  mid-turn); "as of last turn" is the honest semantics - no forced refresh turn.
- The context "output" row sums output + reasoning (cumulative). Fine as a single
  "produced" figure; could split later if wanted.

## Verdict

APPROVE. The sidebar now shows a truthful per-session context block (window fill
%, token mix, turn/tool counts) and a weekly-usage meter (used %, resets, plan),
both hiding cleanly when absent. The review caught and fixed a real over-counting
bug in the context %; verified against real multi-turn sessions.
