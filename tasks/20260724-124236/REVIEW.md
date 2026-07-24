# Review: Route orchestrator session endpoints through the backend

- TASK: 20260724-124236
- BRANCH: fix/backend-agnostic-session-endpoints

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Check suite (out-of-context reviewer, re-confirmed in-session): pytest 440
passed, ruff clean, mypy clean. All seven named DoD tests pass; the
`resolve_codex_home` grep shows only the import and the usage/memory/account
endpoints - none of the four session endpoints. A/B (in-session): reverting
`get_session_transcript` to a codex-home read turned
`test_orchestrator_transcript_uses_backend` red; restoring it green.

Independently re-derived in-session (not adopted wholesale): no session endpoint
raises on a missing/foreign id - `read_transcript` returns `[]`, `read_context`
returns `None` (codex None; claude/opencode map a `None` status via
`_context_from_status`), fork degrades to the edited text, and `delete_session`
returns `False` (codex/claude/mock never raise; opencode catches every
`OpencodeError` -> False and still closes the client in `finally`). The
sync-reads/async-delete asymmetry is justified (opencode delete is a daemon write
on the async `OpencodeClient` boundary; the sole caller is the already-`async`
delete route).

No BLOCKER/MAJOR/MINOR/NIT findings. The reviewer noted the diff includes
incidental ruff-formatting reflows of pre-existing lines (cosmetic, pass all
checks). No open `manual:` DoD items. APPROVE.
