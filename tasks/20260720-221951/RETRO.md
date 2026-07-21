# Retro: A4 Agents dashboard page

- TASK: 20260720-221951
- BRANCH: feature/agents-dashboard (landed 5248077)
- REVIEW ROUNDS: 1 out-of-context APPROVE (1 MINOR EventSource guard fixed, 1 NIT) + in-session round 2

## What went well

- Mirroring `projects-view.ts` (pure render + injected actions + the select-race
  guard) made the whole page fast, consistent, and jsdom-testable with no new
  patterns to invent.
- Keeping the SSE `EventSource` strictly inside `startAgents` (out of the pure
  render) kept the render side-effect-free, so the 8 jsdom tests are clean and
  the stream is proven separately by the e2e serve.
- The e2e serve (create project -> create agent -> run -> status=done through the
  real backend + built bundle) proved the entire A0-A4 stack composes - the
  first time the vision ran end to end from the UI down.

## What went wrong

- The EventSource `onerror` closed the module-level `events` var rather than the
  captured `source`, so a stale source's late error could close a newer stream.
  Narrow window (closeEvents suppresses it in practice), but a real footgun.
  Root cause: I id-guarded `onmessage` but not `onerror`. The reviewer caught it.

## What to improve next time

- When several async callbacks capture a shared "current" handle (EventSource,
  a fetch, a timer), guard EVERY callback on identity (`if (current === mine)`),
  not just the obvious one. I guarded the message path and forgot the error path.

## Action items

- [x] MINOR fixed (identity-guarded onerror); NIT (escaping a className) left as
  harmless-and-cheap.
- No new ledger entry: "id-guard all callbacks sharing a handle" is a variant of
  the existing select-race pattern already captured by projects-view; kept here.
