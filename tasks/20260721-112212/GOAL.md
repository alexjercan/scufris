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
- [ ] 20260721-112430 (p48, B2) permission modes (manual|edit|auto) replacing write_enabled [dep: B1]
- [ ] 20260721-112432 (p46, B3) agent description + retire the required goal [dep: B1]
- [ ] 20260721-112433 (p44, F1) SPA dynamic routing + fallback + agent-detail page shell
- [ ] 20260721-112434 (p42, F2) agents as cards + friendly labels + card->page nav [dep: B1, F1]
- [ ] 20260721-112435 (p40, F3) /agents/<id> detail page + per-agent settings-edit [dep: F1, B2, B3]
- [ ] 20260721-112436 (p38, B4) per-agent chat endpoint (message->stream, resume session) + transcript [dep: B2, B3]
- [ ] 20260721-112438 (p36, F4) per-agent chat UI on the detail page [dep: F1, F3, B4]
- [ ] 20260721-112439 (p34, B5) orchestrator as a reserved default agent (multi-session) [dep: F4]
- [ ] 20260721-112440 (p32, B6) sesh.py discovery + Projects discovery/create (no tmux)

## Manual acceptance (batched for the user at Finish)

Accumulates `manual:` DoD items as tasks land; presented at Finish.
