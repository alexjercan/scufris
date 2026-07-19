# Chat UI: token-by-token text, thinking section, live event feed

- STATUS: OPEN
- PRIORITY: 35
- TAGS: feature, agent, ui, spike

## Goal

Render the new streaming events from the app-server backend so the user sees a
turn unfold live: the assistant bubble fills in TOKEN BY TOKEN from `text-delta`
events (markdown re-rendered as it grows); a collapsible/live "thinking" section
that streams reasoning deltas; and a live event feed of tool calls / plan updates
/ process output as they arrive. On done, finalize to the stored message + meta.

## Notes

- Spike: tasks/20260720-002611/SPIKE.md.
- Depends on tatr 20260720-002619 (the app-server backend + SSE event kinds).
- Build on the existing SSE consumer (`sendChatStream`/`parseSseFrames`) and the
  markdown renderer (re-render the growing text; consider debouncing re-render for
  performance on fast token streams). Keep the reasoning section visually distinct
  from the answer and collapsible (it can be long).
- Escape everything; render side-effect-free where practical for jsdom; keep the
  non-app-server (exec) path working with its existing tool-chip + timer UI.
