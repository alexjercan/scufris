# REVIEW: T6 Telegram live turn streaming

- TASK: 20260726-201901
- ROUND: 1 (out-of-context reviewer, general-purpose agent)
- VERDICT: APPROVE (with recommended follow-ups)

## Summary

State machine is correct: chronological message-per-phase ordering holds,
force-flush on phase boundaries prevents lost reasoning tails, all dynamic HTML
text is escaped, the final answer stays plain, error mapping never leaks raw
backend detail, the bus subscription is not leaked (traced through `subscribe`'s
finally + the supervisor's bus-close), and the change is cleanly additive to the
web SSE consumer. No blockers or majors affecting the common path.

## Findings and dispositions

- #7 [major, test quality] Throttle SUPPRESSION and the unchanged-body guard were
  untested (DoD #2). FIXED: added `test_reasoning_edits_are_throttled` (large
  interval + several deltas -> one thinking send + a single force-flush edit on
  done) and `test_unchanged_reasoning_is_not_re_edited` (a no-op delta produces no
  edit even at interval 0).
- #1 [minor] 4096 cap not airtight under `html.escape` expansion (trim happened
  before escaping). FIXED: `_format_reasoning` now escapes FIRST then tail-trims
  the escaped body, so `len(body)` is truly bounded; added
  `test_format_reasoning_caps_length_after_escaping` with escapable chars.
- #5 [minor] `last_body`/`last_edit` advanced even when an edit failed. FIXED:
  `_edit_message` returns a bool; `flush_reasoning` advances the throttle clock
  always (so a failed edit is not retried faster than the interval) but only marks
  the body delivered on success (a dropped edit re-attempts at the next change).
- #8 [minor, test quality] Post-tool bubble reset was proven only indirectly.
  FIXED: added `test_post_tool_reasoning_edits_the_new_bubble` (reasoning -> tool
  -> reasoning x2 -> the edit targets the SECOND message_id).
- #2 [minor] A whitespace-only first reasoning delta can leave a bare "Thinking..."
  bubble. ACCEPTED for v1 (cosmetic; chronological order holds).
- #3, #4, #6, #9, #10 [nit/confirmation] No action: escaping story solid, final
  answer plain by design, message_id=None degradation reasonable, no web-SSE
  regression, stream=False error path correct.
