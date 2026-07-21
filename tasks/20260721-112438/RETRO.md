# Retro: F4 per-agent chat UI on the detail page

- TASK: 20260721-112438
- BRANCH: feature/agent-chat-ui
- REVIEW ROUNDS: 1 (out-of-context APPROVE, 1 NIT addressed)

## What went well

- Reading `agent-view.ts` (1300 lines) BEFORE planning revealed that the
  "reuse the chat helpers" framing in the task was only half right: the truly
  reusable piece is the STREAMING (parseSseFrames + the SSE consume loop), while
  the render/composer is welded to agent-view's module globals (sessions
  sidebar, fork/edit, image attach, slash commands) - none of which the per-agent
  chat needs. So I extracted a shared `chat-stream.ts` and wrote a lean,
  self-contained chat rather than trying to de-globalize the landing chat. That
  kept the diff small and the new component fully testable.
- Spotting the poll-vs-persistence hazard early: `startAgentDetail` polls status
  and `replaceChildren`s `#agent-detail` every 2s. A chat inside that root would
  be wiped mid-conversation. Mounting the chat in a SEPARATE `#agent-chat` root
  sidesteps it entirely - two independent regions, one polled, one persistent.
- Pure `renderChatLog` + injected `{streamTurn, loadTranscript}` deps meant the
  whole send/stream/transcript/disable/XSS flow is jsdom-driven with no fetch,
  including a real mid-flight "disabled while streaming" assertion via a pending
  promise.
- The extraction preserved `agent-view.ts`'s public API (re-export + wrapper),
  so its 54-test suite stayed green untouched - the reviewer confirmed the moved
  code is byte-for-byte identical.

## What went wrong

- Nothing broke. The only miss was cosmetic (R1.1 NIT): `onDone` overwrote the
  assistant text only when `reply.text` was truthy, so a genuinely empty turn
  would render a blank bubble. Root cause: I mirrored the streamed-text-wins
  logic but dropped the landing chat's `... || "(no reply)"` tail. Fixed +
  pinned by a test.

## What to improve next time

- When a task says "reuse component X", first check whether X is actually
  reusable or is welded to module state - extracting the genuinely-shared
  primitive (here the stream) and re-implementing the thin stateful shell is
  often cleaner than de-globalizing a tangled module. Name the split in the plan.
- A persistent widget must NOT live inside a DOM region that a poll
  `replaceChildren`s - give it its own root. Ledgered.

## Action items

- [x] Review APPROVE, NIT addressed in round 1, no follow-ups.
- Milestone 3 (B4 + F4) complete: a real per-agent conversation now works.
  Next: B5 (orchestrator as a reserved default agent) then B6 (sesh discovery).
