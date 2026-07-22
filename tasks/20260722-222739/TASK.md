# T5: Telegram reply rendering + end-to-end example (final-per-turn + tool summary; examples/ script; respx integration test)

- STATUS: OPEN
- PRIORITY: 32
- TAGS: spike,telegram,feature,ui

## Goal

Reply rendering plus an end-to-end proof for the Telegram bot. Render one final
message per turn with a "typing..." chat action while the orchestrator streams,
and a short tool-summary line (full edited-message token streaming is a later
polish). Ship an `examples/` script that boots the bot against a stubbed Bot
API + the mock backend, and an integration test exercising receive-message ->
orchestrator turn -> reply.

## Notes

- Spike: tasks/20260722-221359/SPIKE.md (Q5, rendering).
- Depends on: T4. May merge with T4 at plan time if the two prove inseparable
  (flow allows re-cutting inseparable tasks).
- Harness-first (AGENTS.md): respx-stubbed Telegram + mock backend; the
  `examples/` script doubles as documentation.
- spike-seeded; plan into steps with /plan before /work.
