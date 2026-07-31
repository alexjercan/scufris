# Retro: Condense repository agent guidance

- TASK: 20260731-131543
- BRANCH: master (uncommitted working-tree diff)
- REVIEW ROUNDS: 1

## What went well

- Mapping each long section to an authoritative README kept safety rules while removing duplicated rationale.
- Targeted code greps caught the `login.ts` fetch exception before close-out.
- Out-of-context review found transport and pointer errors hidden by otherwise accurate summaries.

## What went wrong

- The first pointer check covered the README table but missed the lower release link.
- "Operator session only" generalized the HTTP credential rule across Telegram's separate allowlisted credential.
- Root cause: searches followed edited locations, not every reader; security wording named the actor but not the transport boundary.

## What to improve next time

- After moving documentation, grep all live doc surfaces for the old target and anchor.
- State authentication rules as transport, credential, and permitted action.

## Action items

- Bumped `verify-a-doc-citation-by-running-the-grep` to x2.
- Added `scope-auth-rules-to-their-transport` at x1.
- No follow-up code task.
