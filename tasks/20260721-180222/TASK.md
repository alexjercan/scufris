# B5d: converge the landing + per-agent chat UI on one component

- PRIORITY: 31
- TAGS: agents, frontend
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Make the landing page and the per-agent detail page use ONE chat component
(`agent-chat-view.ts`, the F4 lean chat with injected deps + local state),
retiring the globals-heavy landing chat in `agent-view.ts`. The orchestrator's
page ALSO shows the session switcher (B5c) + context/usage boxes; project agents
show the lean chat only. This removes the duplicated chat implementations.

## Feature disposition (user decisions, 2026-07-21)

The landing-only chat features converge as follows:

- **Image attach -> ALL agents.** Port to the shared component; every agent's
  chat can attach an image (backends already accept `image_paths`).
- **Slash commands -> ALL agents.** Port the slash-command menu to the shared
  component for every agent.
- **Export conversation -> ALL agents.** Port the export action to the shared
  component for every agent.
- **Session switcher + context/usage sidebar -> ORCHESTRATOR only** (already
  decided in the goal; project agents are single-session).
- **Fork/edit a message -> ALL agents, but DIFFERENT semantics by session model:**
  - Orchestrator (multi-session): fork branches into a NEW session seeded with
    the prior turns + the edit, PRESERVING the original session (today's
    `/api/agent/session/fork`).
  - Project agent (single-session): fork "reverts back" - it rewinds the ONE
    session to the fork point and continues from the edited message; NO new
    session is kept (the old tail is abandoned). Needs a per-agent fork endpoint
    that reads the agent's transcript, seeds a turn from messages[:index] + the
    edit, and makes that the agent's (sole) session.

## Steps

- [x] Backend: add `POST /api/agents/{id}/fork` (single-session revert-fork).
      Read the agent's transcript via `backend.read_transcript`, build the seed
      with `format_fork_seed(messages[:index], text)`, and launch a seeded turn
      via `_launch_agent_turn` (its done-event session id becomes the agent's
      session - the old tail is dropped). 404 unknown, 422 empty/missing project,
      409 active. Reject fork on the orchestrator here (it keeps the multi-session
      `/api/agent/session/fork`). Test with the mock backend + a seeded transcript.
- [x] Frontend: extract the landing-only pure helpers worth keeping into shared
      modules the one chat component opts into - image attach, slash commands,
      export, markdown/timestamp helpers (much may already be shared from F4).
      Grep agent-chat-view.ts for what it already has; only extract the gaps.
      Done: `chat-format.ts` (fmtTokens/parseIso/formatTimestamp/relativeTime),
      `chat-commands.ts` (SlashCommand/matchSlashCommands/chatMarkdown/download),
      `chat-image.ts` (readImageFile), `chat-sidebar.ts` (renderSessions/
      renderContext/renderUsage), `chat-stream.ts` grew a `streamPost` raw-body
      variant for the fork endpoint.
- [x] Frontend: add fork/edit to the shared `agent-chat-view` with a
      session-model switch: orchestrator -> POST `/api/agent/session/fork` (new
      session, JSON); project agent -> POST `/api/agents/{id}/fork` (revert-in-
      place, SSE). The edit-to-fork affordance renders for both via injected
      `forkTurn`; the confirm-button verb + hint copy differ ("fork" vs "revert").
- [x] Frontend: mount `agent-chat-view` on the landing entry against the
      orchestrator's endpoints + the session switcher + context/usage sidebar.
      The landing IS the orchestrator's page now (index.html reshaped to the
      sidebar + `#agent-chat` mount, mirroring the detail page). Session switcher
      renders ONLY for the orchestrator (project agents pass no sidebar wiring).
- [x] Frontend: retire `agent-view.ts`'s duplicated chat/log/composer/fork - it
      is now a thin orchestrator ENTRY (wires the shared component + sidebar), no
      second chat implementation. Tests ported into per-module test files
      (agent-chat-view/chat-sidebar/chat-format/chat-commands/chat-stream) plus a
      renderAgentPanel + orchestrator-fork test in agent-view.test.ts.

## Definition of Done

- One chat component serves both the orchestrator (landing) and project agents;
  `agent-view.ts` no longer holds a second chat implementation
  (cmd: `grep -rn "renderLog\|sendChatStream" web/src/agent-view.ts` -> gone/thin).
- The orchestrator page shows the session switcher; a project page does not.
- Image attach, slash commands and export work on BOTH a project agent and the
  orchestrator (test: the shared component renders them for a project agent).
- Fork works on both with the right semantics: orchestrator fork keeps the source
  session (new one created); project fork reverts the single session (test:
  `POST /api/agents/{id}/fork` drops the tail and resumes the new session; a
  frontend test that a project agent's edit-to-fork calls the per-agent endpoint
  and the orchestrator's calls the session fork).
- Full web + backend gate green.
- manual: the landing chat looks/feels like before but is the shared component;
  editing a past message on a project agent reverts that conversation in place.

## Status / resume (ALL steps done; ready for review)

Branch `feature/converge-chat-ui`, STATUS IN_PROGRESS - implementation complete,
full web `npm run ci` (format+lint+test+build) green and backend `python -m
pytest` green.

DONE:
- Step 1 (backend): `POST /api/agents/{id}/fork` (single-session revert-fork) +
  `AgentForkRequest`. Reads the agent transcript, seeds from
  `messages[:index] + text`, launches against `agent.model_copy(session_id=None)`
  so the seed opens a fresh session (persist writes it back, dropping the tail);
  streams SSE like `/chat`; orchestrator -> 409.
- Steps 2-5 (frontend convergence). `web/src/agent-chat-view.ts` is now the ONE
  full chat component: `createAgentChat(root, config): ChatControl` builds its own
  DOM + keeps local state, with opt-in capabilities (image attach, slash palette,
  export via a slash command, edit-to-fork). Fork is injected via `config.forkTurn`
  so the SAME edit affordance renders for both, differing only in endpoint + verb/
  hint copy: orchestrator -> `/api/agent/session/fork` (new session, JSON adapter
  that calls handlers.onDone); project agent -> `/api/agents/{id}/fork` (revert,
  SSE via the new `streamPost`). The streaming pending-bubble (working/thinking/
  throttled markdown) + no-yank pill are ported in. `startAgentChat` (per-agent,
  no sidebar) and the rewritten `agent-view.ts` (orchestrator ENTRY: wires the
  component + the sessions/context/usage sidebar) both mount into `#agent-chat`.
  index.html reshaped to sidebar + `#agent-chat` (mirrors agent-detail.html).
- Extracted shared side-effect-free modules: `chat-format.ts`, `chat-commands.ts`,
  `chat-image.ts`, `chat-sidebar.ts`; `chat-stream.ts` gained `streamPost`.
- Tests ported to per-module files (agent-chat-view/chat-sidebar/chat-format/
  chat-commands/chat-stream) + renderAgentPanel & orchestrator-fork in
  agent-view.test.ts. 151 web tests pass. DoD grep
  `grep -rn "renderLog\|sendChatStream" web/src/agent-view.ts` -> empty;
  agent-view.ts is 262 lines (was 1263).
- MANUAL (still to eyeball in the served bundle): visual layout of the landing +
  detail chat, and that editing a project-agent message reverts in place.

LESSONS confirmed: ran the FULL `npm run ci` (the webpack build is the real type
gate - vitest alone does not type-check); build-DOM-not-parse-HTML for untrusted
markdown kept; the no-yank scroll lives inside `createAgentChat`. GOTCHA hit:
`el()` returns HTMLElement so `.disabled` needs a real `document.createElement
("button")` (typed HTMLButtonElement); and eslint `unbound-method` fires when you
extract an interface METHOD into a const - declare those as function-typed
PROPERTIES instead.

## Notes
- Depends on: B5c (20260721-180219). Blocks: B5e (partly).
- Biggest FRONTEND slice; agent-view.ts is ~1300 lines with many landing-only
  features (sessions sidebar, fork/edit, image attach, slash commands, export).
