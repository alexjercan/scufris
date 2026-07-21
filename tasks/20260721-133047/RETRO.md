# Retro: MB1 agent model follows the backend + editable model in settings

- TASK: 20260721-133047
- BRANCH: fix/model-follows-backend
- REVIEW ROUNDS: 1 (out-of-context APPROVE, zero findings)

## What went well

- The bug report named the exact symptom (mock -> claude keeps gpt-5.5), so
  the reproduction was a two-line trace: B1's `default_model_for` runs only in
  `create()`, never in `update()`. Writing the failing test first
  (`test_update_backend_redefaults_model`) aimed the fix instead of guessing.
- Choosing a server-authoritative `GET /api/agents/backends` (ids + labels +
  default models) over hardcoding defaults in the frontend removed a whole
  class of drift: the frontend cannot know `claude_model`/`agent_model`, so any
  frontend-side default would silently diverge from the server's. As a bonus it
  fixed a latent gap - the picker now shows `mock` exactly when the dev flag is
  on, instead of a hardcoded two-item list.
- The `effective backend` framing made the five update() cases collapse into
  three lines (explicit-non-empty wins; blank or omitted-on-switch re-defaults),
  and the reviewer could trace every case against it.
- Reusing the F3 `agentFields` seam meant the model field + auto-fill landed in
  one place and both forms got it for free.

## What went wrong

- Process slip, not a code bug: I set the task STATUS to IN_PROGRESS on the
  MASTER copy of TASK.md (before sprouting), so the branch's copy was still
  OPEN - the edit did not travel. Caught it at close-out when a sed on
  IN_PROGRESS was a no-op. Root cause: I created the task + wrote its plan on
  master (a planning commit), then sprouted, but did the IN_PROGRESS edit in the
  wrong checkout order. No harm (STATUS is bookkeeping) but it wasted a step.

## What to improve next time

- Do the IN_PROGRESS flip as the FIRST edit inside the sprout worktree, after
  `cd`-ing in - never on the main checkout before sprouting. (The work skill
  already says "only after you are in it, set STATUS to IN_PROGRESS"; I planned
  on master first and let the ordering slip.)
- The deeper lesson is general: a value DERIVED from another field at create
  time (a per-backend default, a computed slug, a derived flag) must be
  recomputed on every mutation path that can change its source, not only in
  create() - ledgered below.

## Action items

- [x] Review APPROVE, no follow-ups.
- Next: Milestone 3 = B4 (per-agent chat endpoint) then F4 (chat UI). The model
  field now shows the right model on the detail page the chat will sit on.
