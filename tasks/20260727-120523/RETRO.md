# Retro: Move MCP per-server health into the Health card

- TASK: 20260727-120523
- BRANCH: feature/mcp-health-in-health-section
- REVIEW ROUNDS: 1 (APPROVE, out-of-context, no findings)

What/why in TASK.md/NOTES.md; process only here.

## What went well

- Confirmed the two ambiguous forks (per-server rows vs aggregate; drop the bulb
  vs keep it) up front via AskUserQuestion before writing code, so the build had
  no shape surprise and the review found nothing.
- Reused the existing `mcp_health.probe_server` / `servers_for_audience` verbatim -
  no new probe logic, just a new consumer (`agent_health`) and a UI subtraction.
  Small, focused diff.
- The prior cycle's fresh ledger lessons prevented recurrence: formatted ONLY the
  touched files (no unrelated drift), and `git add`ed the new task dir before the
  flake gate (no stale-tree failure). First-round APPROVE with zero findings.

## What went wrong

- Nothing material. One self-caught slip: the DoD `cmd:` grep was written too
  broad (`health__dot`), which legitimately still lives in `healthRow` (the Health
  card); tightened it to the actually-removed symbols before running. Caught
  during the work verify, not in review.

## What to improve next time

- When a DoD proves an ABSENCE by grepping a shared file, scope the pattern to the
  symbols actually removed, not a class name a sibling renderer still uses
  legitimately - the same "scope the absence-grep" discipline as the ledger's
  `scope-absence-greps-to-the-diff-not-the-file`, applied to a shared-file class.

## Action items

- [x] No new ledger lessons (nothing recurred; the DoD-grep-scope point is already
  covered by `scope-absence-greps-to-the-diff-not-the-file` in the ledger).
