# Decision: what the size guard covers and how its allowlist ratchets

- STATUS: ACCEPTED
- DATE: 2026-07-31
- TASK: 20260731-171420
- TAGS: maintainability, kiss, tooling
- EPIC: 20260731-171411

## Context

Epic 20260731-171411 fixes the cap at 600 lines for source and 900 for tests,
and requires the allowlist to end empty except `scufris/app.py` and
`tests/test_app.py`. Three details are unresolved by that record and are
load-bearing for the guard's implementation.

1. `web/src/style.css` is 2662 lines - the largest file after `app.py` - and no
   child of the epic owns a CSS split. If the guard covers `.css`, the epic
   cannot reach its empty-allowlist criterion.
2. An allowlist can record a per-file line budget or a bare path. The epic
   asserts "entries may only be removed" but names no mechanism.
3. `scufris/hostd/README.md` deliberately links `tasks/*/DECISION.md`. The
   epic's ID-citation grep excludes only `*.json`, so it fails on that README.

## Decision

1. The guard walks `scufris/**/*.py`, `tests/**/*.py`, and `web/src/**/*.ts`
   (with `*.test.ts` taking the 900 cap). `.css`, `.html`, and `.json` are not
   covered.
2. The allowlist is a path-only `frozenset`. Exceeding the cap while
   allowlisted passes; exceeding it unlisted fails; being listed while inside
   the cap also fails, as a stale entry to delete.
3. The ID-citation ban applies to code comments and docstrings. The proof grep
   adds `-g '!*.md'`; Markdown may cite task records.

## Alternatives considered

- Cover `.css` and allowlist it permanently. Rejected: a permanent entry makes
  the allowlist a config knob rather than a ratchet, and the epic's
  empty-allowlist criterion would have to be relitigated.
- Per-file line budgets in the allowlist. Rejected: churns the guard on every
  edit to a large file, and still lets `app.py` sit one line under 3769
  forever. The stale-entry failure gets the ratchet with one `frozenset`.
- Strip the README's task links to satisfy one uniform grep. Rejected: the
  epic's cost is context spent reading code; a navigable pointer in a README is
  not lore in a hot file.

## Consequences

- Stylesheet growth is unguarded. Revisit only if a CSS change ever needs its
  own context budget.
- An allowlisted file may grow without limit until its owning task lands.
  Bounded by the epic finishing.
- Each split child must delete its allowlist entries in the same change, or
  `nix flake check` fails on a stale entry. That is intended.
- Epic DoD 4 should be read with `-g '!*.md'` added.
