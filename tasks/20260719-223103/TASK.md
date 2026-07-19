# Agent chat: live turn progress and streaming feedback

- STATUS: OPEN
- PRIORITY: 38
- TAGS: feature, agent, ui, spike

## Goal

`codex exec` turns can take many seconds to minutes (it reasons and runs tools),
but the pending state is the literal string "..." with no spinner, elapsed time,
tool activity, or cancel - so a slow turn is indistinguishable from a hang.
Replace it with real feedback: a working indicator, an elapsed timer, and live
"running <tool>..." derived from the `codex exec --json` per-item events we
already produce but currently discard until the turn ends.

## Notes

- Spike: tasks/20260719-223054/SPIKE.md (P0). Also see the agent-internals lesson
  `harvest-the-stream-you-already-run` - the item events are already there.
- OPEN QUESTION for /plan (from the spike): `codex exec --json` is TURN-level, not
  token-delta, so we can stream tool start/finish + a live timer, not token-by-
  token text. Choose: (a) an SSE endpoint (e.g. `/api/chat/stream`) that forwards
  the item events for live tool activity + spinner - richer, needs backend; or
  (b) a pure client-side elapsed timer + animated indicator, no backend change -
  cheaper. Decide by effort/value at /plan.
- Consider a cancel/abort affordance for a runaway turn.
