# F4: per-agent chat UI on the detail page (reuse the agent-view chat helpers)

- PRIORITY: 36
- TAGS: agents, frontend
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

The per-agent CHAT UI on the detail page: reuse the pure chat helpers from
`agent-view.ts` (parseSseFrames, sendChatStream, markdown render, no-yank scroll,
composer) de-globalized into `agent-detail.ts`'s own state, targeting the
per-agent chat/transcript endpoints. Multi-turn conversation with the agent,
like the landing page.

## Steps

- [x] Extract the shared streaming helper into `web/src/chat-stream.ts`:
      `parseSseFrames` + `StreamHandlers` + `streamChatTurn(url, message,
      handlers, image?)` (the URL-parameterized form). Point `agent-view.ts` at
      it: re-export `parseSseFrames` and keep `sendChatStream(message, handlers,
      image?)` as a thin wrapper over `streamChatTurn("/api/chat/stream", ...)`
      so its existing tests stay green.
- [x] New `web/src/agent-chat-view.ts` - a self-contained per-agent chat with
      its OWN local state (no module globals): a pure `renderChatLog(log,
      messages)` (assistant via `renderMarkdown` + tool chips, user via
      `textContent`), a composer (textarea + send, Enter-to-send), no-yank
      auto-scroll (local `stickToBottom` + `isNearBottom`), and
      `createAgentChat(root, agentId, deps)` wiring streaming + transcript load
      via INJECTED deps (`streamTurn`, `loadTranscript`) so jsdom tests drive it.
      `startAgentChat()` reads the id from the path and wires the real deps
      (`streamChatTurn` to `/api/agents/<id>/chat`, transcript fetch).
- [x] A send: append the user msg + a streaming assistant msg, disable the
      composer, stream deltas into the live msg (text_delta appends, tool adds a
      chip, done finalizes with the authoritative reply text, error shows), then
      re-enable. Rebuild history from `GET /api/agents/<id>/transcript` on mount.
- [x] `web/src/agent-detail.html`: add a `#agent-chat` container AFTER
      `#agent-detail` (its own root, so the status poll's `replaceChildren` on
      `#agent-detail` never wipes the chat). `web/src/agent-detail.ts`: call
      `startAgentChat()` alongside `startAgentDetail()`.
- [x] `web/src/style.css`: lay out the `#agent-chat` section (reuse the existing
      `chat__msg`/`chat__msg--md`/composer classes for visual consistency).
- [x] Tests: `agent-chat-view.test.ts` (render user/assistant log incl. markdown
      + tool chips; send appends user + streams assistant to done; transcript
      rebuild on mount; composer disabled while streaming; XSS on user text +
      an assistant tool name). Keep `agent-view.test.ts` green through the
      extraction.

## Definition of Done

- The detail page hosts a working multi-turn chat: sending a message streams the
  agent's reply into the log and the composer re-enables on done
  (test: `sends a message and streams the reply`, `disables the composer while
  streaming`).
- History rebuilds from the transcript endpoint on load
  (test: `rebuilds the conversation from the transcript`).
- The chat survives the detail page's status poll (separate `#agent-chat` root)
  (cmd: `grep -n 'agent-chat' web/src/agent-detail.html`).
- Assistant markdown is built safely and user text is escaped
  (test: `escapes hostile user text and assistant tool names`).
- The whole web gate passes (cmd: `npm run ci` in web/).
- manual: hold a multi-turn conversation with an agent in the browser and it
  resumes across turns.

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (recommendation F4; reuse map - the chat components lift, the
  wiring is new).
- Depends on: 20260721-112433 (F1), 20260721-112435 (F3), 20260721-112436 (B4).
- Close-out: the reuse turned out to be mostly the STREAMING helper, not the
  landing chat's render/composer (which is deeply tied to agent-view module
  globals - sessions sidebar, fork/edit, image attach, slash commands, none of
  which F4 needs). So I extracted a shared `chat-stream.ts` (parseSseFrames +
  streamChatTurn(url,...)) used by both, and wrote a lean self-contained chat
  with its OWN local state rather than de-globalizing agent-view's. The key
  structural decision: the chat lives in a SEPARATE `#agent-chat` root, because
  `startAgentDetail`'s status poll `replaceChildren`s `#agent-detail` every 2s -
  a chat inside it would be wiped mid-conversation. Two independent roots on the
  shell keep the polled region and the persistent chat from stepping on each
  other. Pure `renderChatLog` + injected `{streamTurn, loadTranscript}` deps
  made the whole send/stream/transcript/disable/XSS flow jsdom-testable without
  fetch. Bundle-verified the built shell carries `#agent-chat` + the chat wiring;
  the B4 endpoints it calls were e2e-proven; the interactive browser
  conversation is the batched manual check.
