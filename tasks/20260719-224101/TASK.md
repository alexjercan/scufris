# Agent sessions: fork a conversation by editing a message

- PRIORITY: 30
- TAGS: feature, agent, ui, spike
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Implementation

- Backend: `sessions.py` `format_fork_seed(context, text, max_turns)` (pure;
  prior turns as a context preamble + edited text; capped at `FORK_CONTEXT_TURNS`;
  empty context -> just the text). `app.py` `POST /api/agent/session/fork`
  {source_id, message_index, text}: reads the source transcript, seeds from
  `messages[:index]`, `new_session()` + `chat(seed)`, returns `{current, reply}`
  (under chat_lock; 503 disabled).
- Frontend: chat log refactored to a `_messages` source-of-truth + `renderLog`, so
  each message has a stable index; user messages get a keyboard-reachable "edit"
  affordance that opens an inline editor; "fork" POSTs the fork and rebuilds the
  log (kept context + edit + reply). `resetUsage` split from `_resetAgentState` so
  a fork does not wipe its own messages. `style.css` for the editor + edit button.
- Tests: backend `format_fork_seed` (context/empty/cap) + endpoint (seed carries
  prior turns + edit, drops the message after the fork point, 503). jsdom:
  edit-on-user-only, inline-editor-prefill, hostile-message escape. Live-verified:
  fork at index 2 kept the prior exchange, seeded the edit, new session became
  current.

## Goal

ChatGPT-style branching: edit an earlier USER message in a conversation and start
a NEW session from that point, keeping the context BEFORE the edited message and
discarding everything after. The original session is untouched.

## Design constraint (the real fork here)

`codex exec` has NO native "branch a session at turn N" - a codex session is an
append-only rollout we do not control mid-stream. So fork is implemented by
SEEDING a new session: build a prompt from the transcript up to the edited message
(prior user/assistant turns rendered as context) + the edited text, start a fresh
session, and run that as turn 1. codex then carries the pasted context. Honest
limitation: the prior turns enter as user-provided text, not codex's native
history, and a very long history inflates the seed prompt - so CAP the pasted
context (most recent N turns / a char budget) and note it.

Likely surface (for `/plan`): a backend `POST /api/agent/session/fork`
{source_id, message_index, text} that reads `read_transcript`, formats
messages[:message_index] + edited text into a seed prompt, `new_session()` +
`chat(seed)`, returns the reply + new current id; the frontend renders an inline
edit control on past user messages that on submit forks, then shows the
reconstructed history (messages up to the edit + the edit) followed by the reply.

## Notes

- Spike: (none - direct feature request, but the fork-vs-native-branch decision
  above is a genuine design call - captured here so /plan does not re-litigate).
- Builds on the sessions backend + sidebar + transcript endpoint
  (20260719-212203 / 212205). Depends on nothing in this batch but is the most
  complex; flow it LAST.
- Forking runs a real codex turn (subscription cost) - same as any chat.
- Keep render side-effect-free for jsdom; escape everything; a harness-level test
  that a fork seeds a new session and preserves prior context (with a fake runner
  asserting the seed prompt contains the prior turns + edited text).
