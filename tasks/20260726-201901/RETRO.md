# RETRO: T6 Telegram live turn streaming

- TASK: 20260726-201901
- OUTCOME: delivered; review APPROVE (round 1) after recommended fixes applied.

## What this was

Make the Telegram bot stream one orchestrator turn LIVE, message-per-phase: a
live-edited "thinking" bubble (full reasoning), a widget message per tool call,
then the final answer. The streaming events already existed on the run's
EventBus (the web SSE consumer used them); Telegram just drained them. So the
work was: switch the `on_message` seam from "return final string" to "yield
StreamEvents off `bus.subscribe`", and add the Telegram-side render state machine.

## What went well

- The spike had already named this exact deferral ("edited-message token
  streaming is a later polish"), so scoping was fast and the seam (EventBus,
  StreamReasoningDelta/StreamTool/StreamDone) was already there. Reading the
  spike + T5 record first paid off - no new streaming machinery needed.
- Surfacing the rendering fork to the user BEFORE planning (message-per-phase vs
  one-evolving-message vs thinking+answer; full reasoning vs compact) with
  concrete ASCII previews got a crisp decision and avoided building the wrong
  shape. Recorded in DECISION.md.
- Keeping the change additive to the shared StreamEvent types (no new field on
  ToolCall) meant zero risk to the web SSE path - confirmed by the reviewer.
- The example script doubled as the acceptance demo: running it printed the
  actual phased render (brain bubble, wrench+check widget, footered answer),
  which is more convincing than any assertion.

## What went wrong / friction

- First-pass tests proved phase ORDERING but not the throttle SUPPRESSION or the
  unchanged-body guard - both stated in DoD #2. The out-of-context reviewer
  caught it (#7). Fixed by adding a large-interval test (many deltas -> one
  forced edit on the done boundary) and an unchanged-delta test.
- The 4096-char cap was trimmed on the RAW reasoning, but `html.escape` expands
  chars (`<` -> `&lt;`), so an escapable-heavy tail could still exceed 4096. Fixed
  by escaping FIRST then trimming the escaped body; added a test with `<`*5000.
- Two Edit/Write attempts failed with "File has not been read yet" because I had
  Read the file at its MAIN-checkout path during understanding, then tried to
  edit the WORKTREE copy. This is exactly the known lesson
  `edit-from-the-worktree-path-not-the-planning-read`; re-Reading the worktree
  path fixed it. Cost a couple of round-trips.

## Lessons to fold at Finish (/lessons)

- `test-the-throttle-suppresses-not-just-that-edits-happen`: when a live render
  is throttled, a test at interval=0 (forcing every edit) proves ordering but
  would stay green if the throttle were deleted. Add a large-interval test that
  asserts intermediate updates are SUPPRESSED and the tail is force-flushed on
  the phase boundary, plus a no-op-update test for the unchanged guard.
- `cap-message-length-after-escaping-not-before`: for a hard length cap on text
  that will be HTML/entity-escaped, trim AFTER escaping (escape expands up to
  ~6x); trimming the raw text does not bound the final message. Cutting the TAIL
  of escaped text is safe from a bare `&` (the cut only drops a leading `&...`).
