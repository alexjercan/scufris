# Retro: T2 - orchestrator control MCP tools over the local HTTP API

- TASK: 20260722-222722
- BRANCH: feature/orchestrator-control-tools
- REVIEW ROUNDS: 1 (APPROVE, out-of-context; 2 NITs addressed in-session)

See TASK.md for what changed and why; this is process only.

## What went well

- Reading the real request models (`ProjectCreate`/`AgentCreate`/...) and the SSE
  frame format (`_relay_bus_sse`) BEFORE writing the tools meant the bodies and the
  SSE parser matched the endpoints exactly - the reviewer cross-checked every field
  and found no mismatch. Grounding in the code, not a guess, paid off directly.
- respx side-effect handlers asserting exact body-dict equality (not just "a call
  happened") made the tests revert-sensitive; the reviewer confirmed they would fail
  on revert.
- Applied the T1 retro lesson: updated the registration-set assertion in the same
  pass as adding the tools, rather than discovering it via a failing test.

## What went wrong

- Two NITs the review caught were pre-emptable: `import json` repeated inside two
  tool functions (a stray from copying the tatr-tool pattern), and an id
  interpolated into a URL path guarded only by `.strip()`. Root cause: I reached for
  the existing lazy-import habit reflexively and did not think about the id as a URL
  segment boundary. Both cheap to fix once named.

## What to improve next time

- When a value crosses into a new domain (here: a tool arg becoming a URL path
  segment), guard it in THAT domain at the boundary - the AGENTS/work-skill rule
  "a validation gate must check a value's meaning in each domain it crosses". One-off
  here; not yet a recurring ledger lesson.

## Action items

- No new ledger entry (nothing recurred 2+ times). Follow-up already tracked: claude
  backend MCP wiring is a noted SPIKE open question; T3 (prune) is next in the queue.
