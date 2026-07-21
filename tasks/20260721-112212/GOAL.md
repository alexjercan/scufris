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
- [x] 20260721-152728 (p39, F5) agent detail UX reshape: chat-first + stats sidebar (no sessions) + Settings modal [user feedback; dep: F1,F3,F4]
      landed 26916f0; 1 review round (out-of-context APPROVE, 2 NITs addressed); two-pane chat-first layout - left sidebar (header + Status/Context stat boxes + Settings button), chat fills the right column; settings form moved into a modal overlay (separate root, survives the poll); no Sessions box, account box deferred. 164 frontend tests.
- [x] 20260721-152737 (p38, F6) model selection as a per-backend dropdown/autocomplete [user feedback; dep: MB1,F3,F5]
      landed e2e5bc5; 1 review round (out-of-context APPROVE, zero findings); model field is a datalist-backed autocomplete of the backend's models (BackendOption.models + models_for with default-prepend), swaps on backend change, keeps free text. 271 backend + 166 frontend tests.
- [x] 20260721-152746 (p37, CLEAN) drop codex exec mode (app_server-only) + refresh .env.example & README [user feedback]
      landed ac0203e; 1 review round (out-of-context APPROVE, 1 NIT); CodexBackend app_server-only, agent_backend Literal drops exec (+ legacy->app_server coercion validator), exec runners retained for the landing chat (until B5), docs refreshed for Agents v2. 271 backend tests.
- [x] 20260721-112439 (p34, B5a) reserved orchestrator agent record (synthetic, undeletable, no project) [dep: F4] (B5 re-cut into B5a-e - user chose full split)
      landed 3cf829f; 1 review round (out-of-context APPROVE, 1 NIT); synthetic reserved orchestrator in AgentStore (get/list, never in agents.json), undeletable (403), projectless (server cwd), backend/model from settings, in-memory run-state; gets a working single-session per-agent chat now. Editable config deferred to B5b. 274 backend + 168 frontend tests.
- [x] 20260721-183828 (p38, BUG2) codex agent in auto/edit permission mode still runs read-only (sandbox not applied) [user-reported mid-flow]
      landed 8cd8c70; 1 review round (out-of-context APPROVE, no findings). Root cause via diagnostic-first live probes: a fresh `codex app-server` process spawns per turn, and `thread/resume` never re-sent the sandbox, so resumed turns reverted to read-only - an auto/edit agent could only write on turn 1. Fix: `thread/resume` now passes `{threadId, sandbox}`. The approval-policy theory was ruled out by probing, not guessed. Pinned by a logging-fake regression test. Lesson `resume-must-re-send-per-turn-runtime-settings` (inverse of exec's `codex-resume-rejects-sandbox`).
- [x] 20260721-180208 (p33, B5bc) retire the Agent protocol + move orchestrator sessions to the unified model [dep: B5a] (merged B5b+B5c - inseparable: they share CodexCliAgent.current_session_id)
      landed 6b97e68 (net -354 lines); 1 review round (out-of-context APPROVE, 1 MINOR + 2 NITs, all fixed). Retired the Agent protocol + CodexCliAgent/AgentHandle/build_agent/MockAgent/DisabledAgent; the landing orchestrator now runs through get_backend().stream() + the supervisor like any agent, session state in AgentStore. Backend-switch clears the orchestrator session (kept the cross-backend-stale-session fix). Fixed a fork self-deadlock (holding serialized(ORCHESTRATOR_ID) while _launch_agent_turn reserves the same key). Backend + web (168) green. Lesson `serialize-then-launch-self-deadlocks-on-shared-key`.
- [x] 20260721-180219 (p32, B5c) orchestrator multi-session -> CLOSED, merged into B5bc (no code shipped)
- [x] 20260721-180222 (p31, B5d) converge landing + per-agent chat UI on one component [dep: B5bc]
      landed 75607f9; 1 review round (out-of-context APPROVE, 1 NIT adopted + 1 NIT recorded). ONE chat component (agent-chat-view: createAgentChat(root, config) with opt-in image/slash/export/edit-to-fork); agent-view.ts 1263->262 lines (pure orchestrator entry wiring the component + sessions/context/usage sidebar); index.html reshaped to sidebar + #agent-chat. Fork injected via config.forkTurn: orchestrator -> /api/agent/session/fork (new session, JSON); project -> /api/agents/{id}/fork (revert, SSE via new streamPost). Extracted chat-format/chat-commands/chat-image/chat-sidebar. 151 web + backend pytest green. manual DoD (eyeball landing/detail layout; project-agent revert-in-place) pending. Lessons: el-helper-returns-htmlelement-not-the-subtype, interface-method-shorthand-trips-unbound-method.
- [x] 20260721-180224 (p30, B5e) retire codex-exec runner + fix settings backend picker [dep: B5b, B5d]
      landed c5bc8e7; 1 review round (out-of-context APPROVE, 2 NITs adopted). Retired the dead turn-level codex-exec runners + orphaned helpers (net -664/+382); the codex app-server runner is the sole survivor. WIDENED SCUFRIS_AGENT_BACKEND to canonical codex|claude|mock (default codex) so the landing orchestrator can run on Claude, keeping the legacy app_server|exec->codex coercion as a load guard while the API input stays strict (raw app_server PATCH -> 422). Health probes the selected backend. Settings picker is server-authoritative (Codex/Claude from /api/agents/backends, Mock only behind the dev flag). Backend ruff+mypy+pytest + web npm run ci green. Lessons: retire-a-path-map-callgraph-and-reroute-shared-tests; bumped narrowing-a-persisted-enum-needs-a-coercion-validator to x2.
- [x] 20260721-112440 (p25, B6) sesh.py discovery + Projects discovery/create (no tmux)
      landed 3132cce; 1 review round (out-of-context APPROVE, 1 MINOR + 1 NIT adopted). New scufris/sesh.py: discover() scans configurable base dirs one level deep -> {path,name,language} (language from marker files), create() mkdirs under a base with NO tmux/subprocess + rejects traversal. config.project_base_dirs (default sesh set; env SCUFRIS_PROJECT_BASE_DIRS). GET /api/projects/discovered = discovered UNION registered (+ base_dirs); POST /api/projects/new mkdirs under an allowed base then registers (422 outside the set/unsafe name, 403 read-only before mkdir). Projects page lists discovered+registered, badges registered, register/create actions. Backend + web suites green. Lessons: guard-a-contract-by-capability-not-source-text, new-config-field-updates-all-its-surfaces.
- [x] 20260721-152749 (p20, ENUM) use enums/Pydantic for stringly-typed options (refactor, do last) [user feedback]
      landed e99aba4; 1 review round (out-of-context APPROVE, 1 NIT adopted). New scufris/enums.py with StrEnums AuthMode/Backend/PermissionMode/AgentState/RunPhase, wired into config, agent_store (record state/permission_mode + mark_*), supervisor (RunState.state, was the untyped RunLifecycle=str), and the app models. Behavior-preserving (StrEnum==string, whole pre-existing suite passes unchanged, no pydantic serialize warnings); mark_finished coerces a str state past model_copy's non-validation. tests/test_enums.py pins membership rejection + string round-trip. Lessons: strenum-field-needs-coercion-on-unvalidated-writes, tightening-a-type-strands-its-type-ignore.

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
- (pending) F5 20260721-152728: `/agents/<id>` opens chat-first with a stats
  sidebar (Status + Context, no Sessions) and Settings behind a button (modal).
  (render + bundle verified; the live layout/feel is user-eyeballed.)
- (pending) F6 20260721-152737: the model field shows the selected backend's
  models as a dropdown/typeahead and still accepts a custom typed model.
  (endpoint + datalist verified; the live dropdown is user-eyeballed.)
- (pending) B5a 20260721-112439: the orchestrator appears on /agents (first),
  opens its page, and cannot be deleted. (API verified live; the card/page is
  user-eyeballed.)
- (pending) B5bc 20260721-180208: hold a multi-turn landing/orchestrator
  conversation and switch sessions - it all still works end to end after the
  reroute onto the unified backend path. (both suites green; the live
  multi-turn + session-switch flow is user-eyeballed.)
- (pending) B5d 20260721-180222: the landing chat looks/feels like before but is
  now the shared component, and editing a past message on a PROJECT agent reverts
  that conversation in place (vs the orchestrator branching a new session).
  (151 web + backend suites green; the live layout/feel + revert-in-place is
  user-eyeballed.)
- (pending) B5e 20260721-180224: on the settings page the backend picker shows
  Codex/Claude (+ Mock only when the dev flag is on), and switching the
  orchestrator to Claude actually runs the landing chat on Claude end to end.
  (backend + web suites green; the picker + Claude-orchestrator run is
  user-eyeballed.)
- (pending) B6 20260721-112440: the Projects page lists my real dirs (discovered
  under the base dirs, unioned with the registered ones) and creating/registering
  a project works end to end. (backend + web suites green; the live listing +
  create/register flow is user-eyeballed.)
