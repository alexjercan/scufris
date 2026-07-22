# T4: Telegram transport (httpx long-poll, chat->orchestrator session, auth allowlist, token config, in-process launch)

- STATUS: OPEN
- PRIORITY: 33
- TAGS: spike,telegram,feature,backend

## Goal

Add a thin async httpx long-poll Telegram client. Run a `getUpdates` loop
against the Bot API (no public webhook), map the single chat to the
orchestrator's session (`agent_store.orchestrator_session_id` /
`set_orchestrator_session`), with `/new` resetting the session and `/help`
listing commands. Gate access by an allowlist of chat ids
(`SCUFRIS_TELEGRAM_ALLOWED_CHAT_IDS`); ignore everyone else. Token via
`SCUFRIS_TELEGRAM_BOT_TOKEN`. Launch as a background asyncio task inside the app
process when a token is configured, calling the orchestrator through the SAME
internal path as `/api/chat/stream` (no self-HTTP for the bot).

## Notes

- Spike: tasks/20260722-221359/SPIKE.md (Q5).
- Depends on: T1 (so the orchestrator has its tools); transport itself can be
  built in parallel with T2/T3.
- New settings fields in `config.py` (env_prefix `SCUFRIS_`): bot token +
  allowed chat ids.
- Reference the old `github.com/alexjercan/scufris-bot` for the command set.
- Test: integration test with a respx-stubbed Bot API + stubbed/mock backend
  driving one receive -> turn -> reply cycle.
- spike-seeded; plan into steps with /plan before /work.
