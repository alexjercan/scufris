# Agent chat: multi-line composer with Enter/Shift-Enter

- STATUS: OPEN
- PRIORITY: 30
- TAGS: feature, agent, ui, spike

## Goal

The prompt composer is a single-line `<input>`. You instruct an agent with detail
(multi-paragraph asks, pasted logs), so replace it with an autosizing `<textarea>`
that grows with content (to a max height then scrolls), with Enter = send and
Shift+Enter = newline, and a clear sending/disabled state. Keep the send button.

## Notes

- Spike: tasks/20260719-223054/SPIKE.md (P1).
- Preserve the existing submit path (`sendChat`) and the disabled-while-sending
  behavior; just change the control + key handling. Keep it side-effect-free for
  jsdom where practical.
