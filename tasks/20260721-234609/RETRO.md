# Retro: U2 - per-agent usage/memory/account panel endpoints

- TASK: 20260721-234609
- BRANCH: feature/per-agent-panel-data (landed fae9161)
- REVIEW ROUNDS: 1 (APPROVE, out-of-context, 1 NIT adopted + 1 no-action)

See TASK.md for what/why and REVIEW.md for the findings. Process notes only here.

## What went well

- The plan made the HONEST-SCOPE call up front (usage/memory/account are
  codex-account-level; claude has no reader, so None/empty is correct, not a
  stub) and reused the existing codex readers behind a one-line
  `_agent_is_codex` dispatch - so the diff is tiny and consistent with the
  singular endpoints. The reviewer confirmed no claude reader was missed.
- Reusing `/status` for context (instead of a fourth new endpoint) avoided a
  codex-only `/context` that would have been LESS capable than `/status` for
  claude - the recon caught that read_context is codex-specific while
  backend.read_status is per-backend.

## What went wrong

- R1.1: the account-model assertion was vacuous. The codex agent was created
  with no model, so its effective model EQUALED the global `settings.agent_model`
  default - so `assert acct["model"]` could not distinguish "returns the agent's
  model" (correct) from "returns the global setting" (the bug). Root cause: I set
  the field under test to the same value as the fallback it must not use.

## What to improve next time

- To prove a field returns X and not its fallback/default Y, set X to a value
  DISTINCT from Y - otherwise the assertion passes for both the correct and the
  buggy implementation.

## Action items

- [x] Adopted R1.1 (explicit distinct model on the codex agent).
- [x] Added lesson `assert-a-distinct-value-not-the-default`.
- No follow-up tasks; U3 (unified settings page) is next.
