# Review: F2 render agents as cards + friendly labels + card->page nav

- TASK: 20260721-112434
- BRANCH: feature/agent-cards

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

CI ran green in the worktree (prettier + eslint + vitest 143/143 + webpack
build). Goal delivered, every ticked step real, all four named DoD tests pass
on their stated criteria. Findings below are non-blocking; addressing the two
MINORs + the turns NIT since they are cheap and this diff caused them.

- [ ] R1.1 (MINOR) web/src/agents-view.ts:134-139 - keyboard activation of the
  inner delete button also navigates. The card's keydown handler fires on any
  Enter/Space that bubbles up, with no target guard; the click path is guarded
  by stopPropagation but keydown is separate. A keyboard user tabbing to the
  delete button and pressing Enter deletes AND opens. Guard with
  `if (ev.target !== card) return;` at the top of the keydown handler.
  - Response: fixed - added an `ev.target !== card` guard to the card keydown
    handler so a keydown bubbling from the delete button no longer navigates.
- [ ] R1.2 (MINOR) web/src/style.css - dead CSS orphaned by this rewrite:
  `.agents__item`, `.agents__item--active .agents__name`, `.agents__name`,
  `.agents__meta`, `.agents__status`, `.agents__events`, `.agents__eventline`
  are no longer referenced (this diff removed their consumers). Remove them.
  `.agents__badge*`, `.agents__lastmsg`, `.agents__back` stay (still used by
  agent-detail-view.ts).
  - Response: fixed - removed `.agents`, `.agents__item`,
    `.agents__item--active .agents__name`, `.agents__name`, `.agents__meta`,
    `.agents__check` (already dead), `.agents__status`, `.agents__events`,
    `.agents__eventline`. Kept `.agents__badge*`, `.agents__lastmsg`,
    `.agents__back`.
- [ ] R1.3 (NIT) web/src/agents-view.test.ts:128 - `toContain("2")` for turns is
  substring-lucky (tokens "20 out" already contains "2"). Assert the turns row
  value explicitly or use a turns value that cannot appear elsewhere.
  - Response: fixed - the test now uses turns=7 and asserts the turns row's
    value span equals "7" (was substring `toContain("2")`).
- [ ] R1.4 (NIT) web/src/agents-view.test.ts:205-214 - the XSS test's description
  half is vacuous (description is not rendered on the card). Name half is real.
  Left as-is or trimmed to name-only.
  - Response: fixed - trimmed the test to the name-only assertion it actually
    exercises (renamed to "escapes a hostile agent name...").
- [ ] R1.5 (NIT) web/src/agents-view.ts:51-58 - `escapeHtml(state)` into a
  className is a semantic no-op (el assigns via .className, not innerHTML);
  mirrors the existing agent-detail-view pattern for consistency. Cosmetic;
  left as-is (badge text is correctly set via textContent).
  - Response: acknowledged, left as-is for consistency with agent-detail-view.
