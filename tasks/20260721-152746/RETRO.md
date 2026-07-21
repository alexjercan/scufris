# Retro: drop codex exec mode + refresh docs

- TASK: 20260721-152746
- BRANCH: feature/drop-exec-mode
- REVIEW ROUNDS: 1 (out-of-context APPROVE, 1 NIT addressed)

## What went well

- The investigation-first framing (already in the plan) prevented an over-broad
  cut: the exec MODE (a dead per-agent option + a landing stream selector) is
  removable, but the exec RUNNERS are still the landing non-streaming chat path,
  so they stay. Separating "mode" from "runner" kept the change safe and the
  landing chat intact.
- The legacy-coercion `field_validator(mode="before")` on `agent_backend` is the
  key robustness move: narrowing the Literal would otherwise BRICK startup for
  any persisted/env `SCUFRIS_AGENT_BACKEND=exec`. Coercing it to app_server on
  load (while the API's AgentConfigUpdate rejects a NEW "exec") is the right
  split - legacy loads, but the surface no longer advertises the dropped value.
- Repointing (not deleting) the exec-referencing tests kept coverage honest: the
  cwd/session + permission-mode tests now assert the same behavior via the
  app_server runner the backend actually uses.

## What went wrong

- Nothing broke. A few tests used "exec" merely as "a valid non-default backend
  value"; the coercion validator turned that into app_server, so those had to
  move to "mock"/"app_server" to keep testing what they meant (rollback, rebuild
  triggers). Easy to miss - caught by running the suite, not by inspection.
- One stale historical docstring (R1.1 NIT) survived - a changelog line that read
  like current behavior. Clarified.

## What to improve next time

- When NARROWING a persisted/config enum (dropping a member), add a before-mode
  coercion for the removed value so existing state loads, and keep the API
  input model strict (reject the removed value on new writes). Ledgered - the
  upcoming enum/pydantic refactor task will hit this repeatedly.

## Action items

- [x] Review APPROVE, NIT addressed.
- This completes the user's feedback batch (bug + F5 + F6 + exec-drop/docs).
  Next original-scope task: B5 (orchestrator as a reserved default agent).
