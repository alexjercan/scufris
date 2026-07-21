# Retro: converge the landing + per-agent chat UI on one component

- TASK: 20260721-180222
- BRANCH: feature/converge-chat-ui (landed 75607f9)
- REVIEW ROUNDS: 1 (APPROVE, out-of-context)

See TASK.md for what/why and REVIEW.md for the findings. Process notes only here.

## What went well

- The convergence direction from B5bc's recon paid off: the F4 lean component was
  the ONE to keep, and the globals-heavy landing chat folded INTO it rather than
  the reverse. `agent-view.ts` dropped 1263 -> 262 lines with no second chat impl,
  and the same `createAgentChat` now serves both pages. This is the follow-through
  on `reuse-the-shared-primitive-not-the-globalized-shell`.
- Injecting fork as `config.forkTurn` (a `(index, text, handlers) => Promise`
  seam) let ONE edit affordance render for both semantics; the JSON orchestrator
  fork and the SSE project fork are just two adapters onto the same StreamHandlers
  contract, so the component never branches on "which page am I".
- The gates were honest: full `npm run ci` (not just vitest) + `nix develop python
  -m pytest`, both re-run by the out-of-context reviewer. Round 1 was APPROVE with
  one cheap NIT, adopted.
- Out-of-context review earned its keep the usual way: it flagged a
  docstring-advertised backend branch ("422 missing project") that had NO test,
  even though every existing fork test passed. Pinned it in one commit.

## What went wrong

- Two type/lint gotchas cost a tsc/eslint round each (caught pre-commit, not in
  review):
  1. `el("button", ...)` returns `HTMLElement`, so `.disabled` does not exist -
     the attach/send buttons needed a real `document.createElement("button")`
     (typed `HTMLButtonElement`). Root cause: reached for the terse `el()` helper
     out of habit for an element whose subtype-specific property I then used.
  2. eslint `@typescript-eslint/unbound-method` fired when I extracted an
     interface METHOD (`forkTurn?(...)`, `onEdit?(...)` shorthand) into a `const`.
     Root cause: method-shorthand members are treated as unbound methods;
     function-typed PROPERTIES (`forkTurn?: (...) => ...`) are not.
- A small chicken-and-egg: slash commands need the control handle, but the handle
  is only returned after the component builds. Solved cleanly with a late-bind
  `control.setSlashCommands([...])` call, but I designed it twice before landing
  on that.

## What to improve next time

- When a built element's subtype property (`.disabled`, `.value`, `.files`) will
  be touched, create it with `document.createElement(tag)` for the precise type;
  reserve the `el()` helper for plain container/text nodes.
- Declare callback members of a config/deps interface as function-typed
  PROPERTIES, not method shorthand, so extracting them into a local never trips
  `unbound-method`.

## Action items

- [x] Adopted R1.1: pinned the 422-missing-project fork boundary (test).
- No follow-up tatr tasks: the two NITs are resolved / recorded; manual DoD
  eyeballing remains the user's acceptance gate.
