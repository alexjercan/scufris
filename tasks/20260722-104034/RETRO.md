# Retro: backend-aware health (claude agents no longer show codex diagnostics)

- TASK: 20260722-104034
- BRANCH: feature/backend-aware-health (landed deef230)
- REVIEW ROUNDS: 1 (out-of-context APPROVE, one MINOR adopted)

See TASK.md for what/why (incl. the folded-in spike findings) and REVIEW.md for
the findings. Process notes only here.

## What went well

- The inline recon WAS the spike: I traced the whole path (settings page fetch ->
  global `/api/agent/health` -> `agent_health(settings)` probing the server
  backend) before writing a line, and wrote the map into TASK.md. That turned a
  vaguely-scoped "make it backend-aware" bug into one precise root cause + a
  four-line surface audit, so the fix was small and confident. No separate SPIKE
  task was needed - I said so in the plan rather than skipping the thinking.
- `agent_health` already branched by backend; the fix was to parametrize the
  input, not to add probe logic. Recognizing that kept the diff to a param + a
  field rename + one endpoint, instead of a rewrite.
- I stated the account-panel `auth_mode` boundary UP FRONT in the DoD (out of
  scope, why, deferred), so the reviewer's account-panel NIT was already
  accounted-for rather than a surprise, and it is now a named follow-up.

## What went wrong

- The MINOR: my backend tests proved the branch SELECTION (claude checks present,
  codex absent) but every case used a missing bin, so `backend_version` was always
  None - I never proved the neutral field carries the RIGHT backend's version.
  I asserted the negative (no codex field) but not the positive (claude version
  present). Root cause: I tested the visible symptom (which checks appear) and
  under-tested the new data field I introduced.

## What to improve next time

- When a change RENAMES/REPLACES a data field (here codex_version ->
  backend_version), add at least one test that the new field is POPULATED with a
  real value on the happy path, not only that the old name is gone and the null
  case works. A field worth adding is worth one positive assertion.
  (Lesson: assert-a-renamed-field-is-populated-not-just-absent.)

## Action items

- [x] Adopted the MINOR (fake executable claude bin emitting a version; assert
      backend_version holds it while backend == claude).
- [x] Add lesson `assert-a-renamed-field-is-populated-not-just-absent`.
- [ ] Follow-up (surfaced, not filed as its own task yet): the account panel's
      auth_mode is still codex-flavored; dispatch it by backend when claude auth
      is modeled. Noted in GOAL-less backlog / this RETRO + REVIEW.
