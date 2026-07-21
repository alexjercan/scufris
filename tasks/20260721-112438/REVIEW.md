# Review: F4 per-agent chat UI on the detail page

- TASK: 20260721-112438
- BRANCH: feature/agent-chat-ui

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Full web gate ran green in the worktree (prettier + eslint + vitest 161/161
across 10 files + webpack build; built shell carries `#agent-chat` and the
bundle has the chat wiring). One non-blocking NIT, addressed below.

Verified by the reviewer:
- `chat-stream.ts` is a byte-for-byte extraction of `parseSseFrames` +
  `streamChatTurn(url,...)`; `agent-view.ts` re-exports `parseSseFrames` and
  keeps `sendChatStream` as a one-line wrapper. `agent-view.test.ts` is
  unchanged and still exercises the real code through the wrapper. Now-unused
  `StreamEvent`/`ToolCall` imports were dropped. Landing behavior unchanged.
- The send flow appends user + streaming-assistant msgs, disables the composer,
  streams text_delta/tool/done/error into the live msg, and re-enables in a
  `.finally()` (fires on resolve AND reject); the `streaming` guard blocks a
  concurrent send; `onError` keeps the UI usable.
- XSS-safe: user text + tool chips via `textContent`, assistant via
  `renderMarkdown` (no innerHTML); the only el()-with-html is a static literal.
- Separate `#agent-chat` root means the status poll's `replaceChildren` on
  `#agent-detail` cannot wipe the chat; single mount, no DOM contention.
- Tests are meaningful (mid-flight disabled assertion via a pending promise,
  reply reaches the log, transcript rebuild, empty no-op, Enter/Shift+Enter,
  XSS); none weakened vs master.

- [x] R1.1 (NIT) web/src/agent-chat-view.ts:154 - `onDone` only overwrites
  `assistant.text` when `reply.text` is truthy, so a genuinely empty turn shows
  a blank bubble instead of a placeholder.
  - Response: fixed - `onDone` now sets
    `assistant.text = reply.text || assistant.text || "(no reply)"`, matching the
    landing chat's placeholder intent. Pinned by a new test.
