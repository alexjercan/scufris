# B5d: converge the landing + per-agent chat UI on one component

- STATUS: OPEN
- PRIORITY: 31
- TAGS: agents,frontend

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

- [ ] Backend: add `POST /api/agents/{id}/fork` (single-session revert-fork).
      Read the agent's transcript via `backend.read_transcript`, build the seed
      with `format_fork_seed(messages[:index], text)`, and launch a seeded turn
      via `_launch_agent_turn` (its done-event session id becomes the agent's
      session - the old tail is dropped). 404 unknown, 422 empty/missing project,
      409 active. Reject fork on the orchestrator here (it keeps the multi-session
      `/api/agent/session/fork`). Test with the mock backend + a seeded transcript.
- [ ] Frontend: extract the landing-only pure helpers worth keeping into shared
      modules the one chat component opts into - image attach, slash commands,
      export, markdown/timestamp helpers (much may already be shared from F4).
      Grep agent-chat-view.ts for what it already has; only extract the gaps.
- [ ] Frontend: add fork/edit to the shared `agent-chat-view` with a
      session-model switch: orchestrator -> POST `/api/agent/session/fork` (new
      session); project agent -> POST `/api/agents/{id}/fork` (revert-in-place).
      The edit-to-fork affordance renders for both; the endpoint + "new session
      vs revert" copy differ.
- [ ] Frontend: mount `agent-chat-view` on the landing entry against the
      orchestrator's endpoints + the session switcher + context/usage sidebar
      (reuse the F5 sidebar boxes). The landing IS the orchestrator's page now.
      Session switcher renders ONLY for the orchestrator.
- [ ] Frontend: retire `agent-view.ts`'s duplicated chat/log/composer/fork once
      the landing uses the shared component. Port its meaningful tests to the
      shared component's test file; drop the dead ones.

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

## Notes
- Depends on: B5c (20260721-180219). Blocks: B5e (partly).
- Biggest FRONTEND slice; agent-view.ts is ~1300 lines with many landing-only
  features (sessions sidebar, fork/edit, image attach, slash commands, export).
  Decide per-feature: port to shared, keep orchestrator-only, or drop.
