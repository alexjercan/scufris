# Add research provenance citations and export UI

- STATUS: OPEN
- PRIORITY: 0
- TAGS: feature,backlog,research,artifacts,frontend
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT

## Story

As a research user, I want to watch progress and inspect the evidence behind a
report, so that I can intervene, verify citations, understand disagreement, and
export a useful artifact.

## Steps

- [ ] Build a research-run view with plan/questions, stage and agent progress,
      remaining budgets, cancellation, failures, and final status.
- [ ] Add evidence/source browsing with search/filter, provenance, researcher,
      freshness, confidence, excerpt limits, and original artifact links.
- [ ] Show claim support, contradiction, missing evidence, skeptic findings,
      deduplication, and quality-gate state without flattening them into chat.
- [ ] Render the final report with stable citation links that open the
      referenced web/PDF/text artifact at the cited location.
- [ ] Add Markdown and structured JSON export containing report, bibliography,
      evidence ledger, conflicts, budgets, and run provenance.
- [ ] Support live updates, reconnect/resume, direct URLs, keyboard operation,
      mobile layout, and large evidence sets.
- [ ] Add browser journeys for successful, conflicting, partial, cancelled, and
      budget-exhausted research runs.

## Definition of Done

- The full mock research run can be observed, cancelled, resumed, and exported
  through visible controls (test: `research-run.spec.ts`).
- Every final citation opens the exact source artifact/anchor
  (test: `research-citation-navigation.spec.ts`).
- Contradictions and unsupported claims remain visible in the final view/export
  (test: `research-conflict-visibility.spec.ts`).
- Large evidence sets remain searchable and responsive on desktop and mobile
  (test: `research-large-ledger.spec.ts`).
- A real exported report is suitable for sharing without separately
  reconstructing its sources (manual: user check).

## Notes

- Epic: 20260729-102218.
- Depends on: 20260729-102221, 20260729-102212, 20260729-102214, and
  20260729-102152.
- Keep the primary research surface evidence-oriented, not a decorative agent
  chat transcript.
