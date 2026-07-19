# Agent sidebar: grouped, labeled sections (sessions / this session / account)

- STATUS: OPEN
- PRIORITY: 30
- TAGS: feature, agent, ui, spike

## Goal

The sidebar reads as one undifferentiated column: the session list scrolls and
drags the context block + weekly meter with it, and nothing frames the three
distinct concerns. Group them into labeled, separately-behaving boxes (the user's
own example):

- **Sessions** - the chat history, in its own scroll area with a visible heading.
- **This session** - the context block (window fill %, tokens, turns/tools).
- **Account** - the weekly-usage meter.

So the history scroll never moves the stats, and each box says what it is. Also:
dedupe/relocate the cryptic head `ctx X · Y out` indicator (now redundant with
the context box), add a one-line explanation or tooltip per stat, and label the
usage "as of last turn" (codex only reports it mid-turn).

## Notes

- Spike: tasks/20260719-223054/SPIKE.md (P1) - this is the user's headline
  example.
- Consumes the existing `/api/agent/sessions|context|usage` endpoints; frontend +
  CSS only. Keep render side-effect-free for jsdom; escape session titles.
- Consider collapsible sections if vertical space is tight; keep the fixed-foot
  behavior so the stats stay visible.
