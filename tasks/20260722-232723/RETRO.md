# Retro: CRUD control MCP tools for projects and agents

- TASK: 20260722-232723
- BRANCH: feature/crud-control-tools
- REVIEW ROUNDS: 1 (APPROVE, out-of-context, no findings)

See TASK.md for what changed and why; this is process only.

## What went well

- The compounding paid off: this task reused T2's `_api_call` contract, `_clean_id`
  guard, and the respx side-effect test pattern verbatim, and applied prior retros'
  lessons up front (registration-set updated in the same pass; the reject guard
  pinned by a NO-http-call test). Result was a clean round-1 APPROVE with zero
  findings - the first no-finding review of the session.
- The user's "regular agents only" scope became one small `_reject_orchestrator`
  helper applied before any HTTP call, and a test that proves no call is made.
- Reading the request models (`ProjectUpdate`/`AgentUpdate`, both `extra="forbid"`)
  first drove the `_provided` "omit None, keep empty-string" semantics correctly on
  the first try.

## What went wrong

- Nothing of substance. The scope shifted mid-request (from a single `update_agent`
  tool to full CRUD for projects+agents) - handled by broadening the tatr task before
  building rather than growing the diff ad hoc, which kept the task record honest.

## What to improve next time

- Keep doing the up-front convention-reuse pass; it is what turned a 5-tool addition
  into a no-finding review. No new lesson - this is the existing pattern working.

## Action items

- No new ledger entry, no follow-ups. The control-tool surface is now full CRUD;
  a `delete`/confirmation UX and the claude-backend MCP wiring remain the only known
  gaps (both already noted against the Telegram spike / its open questions).
