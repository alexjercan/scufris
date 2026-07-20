# Retro: steer the agent to prefer the scufris MCP tools

- DATE: 20260720
- VERDICT: shipped

## What went well

- The bug playbook worked exactly as intended: reproduce first (a live probe that
  showed 16 shell / 0 MCP), then bisect the mechanism with controlled probes rather
  than guessing. The table of five levers in TASK.md is the durable result - a
  future session never has to re-discover that codex ignores instructions files.
- Stripping the preamble at the READ boundary (`read_transcript`, `_read_head`)
  rather than at write time made the fork seed clean for free and kept titles/
  history honest. Putting `STEERING_PREAMBLE` and `strip_steering` in the same
  module means the format and its inverse cannot drift.
- Two-level verification: deterministic unit tests for the wiring, and a real-code
  e2e probe (not a hand-built command) for the actual LLM behavior. The LLM part is
  proven by evidence, never faked in a unit test.

## What went wrong / friction

- Burned a 3-minute tool timeout on the first probe because `codex exec` blocks
  reading stdin when the prompt is an arg but stdin is not closed - the app closes
  it, my probe did not. Fixed with `</dev/null` and running probes in the
  background so a slow model turn never trips the foreground timeout.
- Spent two probes (tool descriptions, then instructions file/AGENTS.md) on levers
  that turned out to be duds. Not wasted - that IS the finding - but I could have
  gone to the prompt-preamble hypothesis sooner given how strong codex's shell bias
  looked in the very first baseline ("I'll gather ... from the shell").

## Lessons

- `codex-tool-choice-only-steers-via-the-turn-prompt` - to make codex prefer an
  MCP tool over its shell, the instruction MUST be on the turn prompt. Probed live
  (0.142.2): strengthened tool descriptions, `-c experimental_instructions_file`,
  and an `AGENTS.md` via `-C` ALL left it at 0 MCP calls / ~all shell; only a
  preamble prepended to the prompt flipped it to 0 shell / 3 MCP. If the preamble
  must stay out of the visible transcript, sentinel-wrap it and strip on read.
- `close-stdin-when-probing-codex-exec-with-an-arg-prompt` - `codex exec "<prompt>"`
  still blocks on stdin ("Reading additional input from stdin...") unless stdin is
  closed; pass `</dev/null`. Run live codex probes in the background (they take
  1-3 min) so a slow turn does not hit the foreground command timeout.

## Follow-ups

- The steering adds ~90 tokens/turn to context. If context pressure ever matters,
  consider first-turn-only steering (accepting some reliability loss) or a shorter
  preamble. Not worth it now.
- Sibling round-2 tasks still open: 102600 (chat head), 102601 (settings/tools
  view), 102602 (discoverability polish).
