# EPIC: Agents UX v2 - cards, per-agent chat page, permission modes, sesh discovery

- DATE: 20260721
- UMBRELLA TASK: 20260721-112212
- LANDING SCOPE: squash-merge each task to `master` (local default), do NOT
  push (user's call). Standard flow landing.

## Goal

Reshape the multi-agent orchestrator (v1, landed) into the UX the operator
actually wants: agents rendered as cards (like the Stats host cards), each
opening a dedicated `/agents/<id>` page that hosts a real chat interface (like
the landing page) plus a small per-agent settings view. Agents stop being a
one-shot "goal run" and become a chattable, project-bound entity. Along the way:
clean up the backend model surface (user-facing "Codex"/"Claude" only; `mock`
dev-only; drop `exec`), replace the write boolean with Claude-style permission
modes (manual/edit/auto, default manual), fix the per-backend model default, add
an optional description, unify the landing orchestrator as a special default
agent, and make Projects discover directories the way `sesh` does.

Direction and feasibility were reviewed in this EPIC's SPIKE.md (the design
decisions are pinned there); this GOAL.md holds the task queue the flow drives.

## Done means

1. User-facing backends are "Codex" and "Claude" (friendly names); `Codex` maps
   to the app_server runner, `exec` is dropped, `mock` is dev-only behind a flag.
   (test: get_backend + label mapping)
2. An agent's write posture is a permission MODE (manual|edit|auto, default
   manual), mapped per backend, replacing the write_enabled boolean everywhere.
   (test: mode -> per-backend flags)
3. An agent has an optional `description`; "goal" is no longer a required run
   input - work is driven by chatting. (test: create without a goal; description
   round-trips)
4. A claude agent shows a claude-appropriate model, not "gpt-5.5". (test:
   per-backend default model)
5. Agents render as cards; clicking one opens a dynamic `/agents/<id>` route
   (real routing + SPA fallback) with detail + a settings-edit form. (manual:
   the page loads and edits persist)
6. The `/agents/<id>` page hosts a multi-turn CHAT with that agent (its own
   session, resumed each turn) over a per-agent chat endpoint. (test: chat
   endpoint streams + resumes; manual: hold a conversation with an agent)
7. The landing orchestrator is a special default agent (undeletable, via the
   backend interface) with multi-session powers; project agents are
   single-session. (manual: orchestrator still works, is not deletable)
8. Projects surface discovered directories (sesh-style scan of the base dirs)
   with inferred metadata, and creating one makes the directory (NO tmux).
   (test: sesh.py scan + create; manual: discovered projects appear)

Overall: the full check suite passes on master (cmd: `nix develop --command
bash -c "ruff check . && mypy . && pytest -q"` plus `npm run ci` in web/), and
`tatr check --ledger LESSONS.md` is clean for this EPIC's tasks.

## Tasks

Updated as tasks land (one line per land). Order = build sequence; deps noted.
Seeded from SPIKE.md; each is coarse (the flow's /plan expands it into steps).

- [x] 20260721-112428 (p52, F0) quick UI polish bugs (SSE reattach on select, status poll interval, empty states)
      landed 1c37cd8; 1 review round (out-of-context APPROVE, 1 NIT); SSE reattach + bounded status interval (focus-guarded) + empty states. 135 frontend tests.
- [x] 20260721-112429 (p50, B1) backend surface cleanup (Codex/Claude only, mock dev-flag, drop exec, per-backend model, labels)
      landed ba8203c; 1 review round (out-of-context APPROVE, 1 MINOR tracked on B5); codex/claude surface + normalize legacy + per-backend model (claude bug fixed) + mock dev-flag. 253 backend + 135 frontend tests.
- [x] 20260721-112430 (p48, B2) permission modes (manual|edit|auto) replacing write_enabled [dep: B1]
      landed f54fc89; 1 review round (out-of-context APPROVE, zero findings); manual|edit|auto replaces write_enabled everywhere, per-backend flags verified live, legacy migration. 255 backend + 135 frontend.
- [x] 20260721-112432 (p46, B3) agent description + retire the required goal [dep: B1]
      landed 8ec92b9; 1 review round (out-of-context APPROVE, 1 NIT deferred to B4); optional description field, goal retired from create UX (kept optional/hidden). 256 backend + 135 frontend.
- [x] 20260721-112433 (p44, F1) SPA dynamic routing + fallback + agent-detail page shell
      landed 4f4a8e1; 1 review round (out-of-context APPROVE, 1 NIT); /agents/<id> serves a detail SPA shell (routes before static mount), agentIdFromPath + read-only renderAgentDetail. e2e-verified. 258 backend + 141 frontend.
- [x] 20260721-112434 (p42, F2) agents as cards + friendly labels + card->page nav [dep: B1, F1]
      landed 30934a0; 1 review round (out-of-context APPROVE, 2 MINOR + 3 NIT addressed); agents render as a .cards grid (name/badge/backend label/project/mode/live turns-tokens), card click -> /agents/<id>, in-page detail/SSE machinery dropped + dead CSS removed. 143 frontend tests.
- [x] 20260721-112435 (p40, F3) /agents/<id> detail page + per-agent settings-edit [dep: F1, B2, B3]
      landed f1e2559; 1 review round (out-of-context APPROVE, zero findings); shared agentFields(context, initial) builder feeds both create + settings forms; detail page swaps read-only backend/desc/mode rows for an editable form that PATCHes /api/agents/{id}; e2e-verified PATCH round-trip. 150 frontend tests.
- [x] 20260721-133047 (p39, MB1) model follows backend: re-default on switch + editable model in settings [dep: F3] (user-reported mid-flow)
      landed e9e2b94; 1 review round (out-of-context APPROVE, zero findings); AgentStore.update re-defaults model to the effective backend, claude default = claude-opus-4-8, new GET /api/agents/backends (server-authoritative picker + defaults), editable model field auto-fills on backend switch. e2e-verified. 262 backend + 153 frontend tests.
- [x] 20260721-112436 (p38, B4) per-agent chat endpoint (message->stream, resume session) + transcript [dep: B2, B3]
      landed 7a316fc; 1 review round (out-of-context APPROVE, zero findings); POST /api/agents/{id}/chat streams a turn (shared _launch_agent_turn with run) + GET /api/agents/{id}/transcript via per-backend read_transcript (pure parse_claude_transcript). 409 test rewritten async after a TestClient-buffering deadlock. e2e-verified. 269 backend + 153 frontend tests.
- [x] 20260721-112438 (p36, F4) per-agent chat UI on the detail page [dep: F1, F3, B4]
      landed bf034c7; 1 review round (out-of-context APPROVE, 1 NIT addressed); self-contained chat in its own #agent-chat root (survives the status poll), shared chat-stream.ts (parseSseFrames + streamChatTurn) reused with the landing chat, streams reply + rebuilds transcript on mount. 162 frontend tests.
- [x] 20260721-152034 (p40, BUG) switching backend leaves a stale cross-backend session; claude resume fails [user-reported mid-flow]
      landed afbeaf8; 1 review round (out-of-context APPROVE, zero findings); root-caused live (claude --resume of an unknown session -> error_during_execution); two-layer fix - AgentStore.update clears session_id on backend change + ClaudeBackend skips --resume for an off-disk session. e2e-verified. 271 backend tests.
- [ ] 20260721-152728 (p39, F5) agent detail UX reshape: chat-first + stats sidebar (no sessions) + Settings modal [user feedback; dep: F1,F3,F4]
- [ ] 20260721-152737 (p38, F6) model selection as a per-backend dropdown/autocomplete [user feedback; dep: MB1,F3,F5]
- [ ] 20260721-152746 (p37, CLEAN) drop codex exec mode (app_server-only) + refresh .env.example & README [user feedback]
- [ ] 20260721-112439 (p34, B5) orchestrator as a reserved default agent (multi-session) [dep: F4]
- [ ] 20260721-112440 (p32, B6) sesh.py discovery + Projects discovery/create (no tmux)
- [ ] 20260721-152749 (p20, ENUM) use enums/Pydantic for stringly-typed options (refactor, do last) [user feedback]

## Manual acceptance (batched for the user at Finish)

Accumulates `manual:` DoD items as tasks land; presented at Finish.

- (pending) F2 20260721-112434: the agents page shows agents as cards; clicking
  a card opens `/agents/<id>`. (e2e proved routing; visual/click is user-eyeballed.)
- (pending) F3 20260721-112435: `/agents/<id>` shows an editable settings form
  and edits persist across a reload. (e2e proved the PATCH slice; the DOM form
  submission is user-eyeballed.)
- (pending) MB1 20260721-133047: in the browser, switching Builder mock -> claude
  updates the model field to claude-opus-4-8 and saving persists it. (e2e proved
  the API re-default; the live dropdown auto-fill is user-eyeballed.)
- (pending) B4+F4 20260721-112436/112438: hold a multi-turn conversation with an
  agent on `/agents/<id>` and it resumes across turns (its own session). (e2e
  proved the /chat stream + /transcript endpoints and the bundle mounts; the
  live browser conversation is user-eyeballed.)
