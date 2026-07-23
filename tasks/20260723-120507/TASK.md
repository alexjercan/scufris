# Harden pending-agents poll onto a collision-proof path (not under /api/agents/{id})

- STATUS: CLOSED
- PRIORITY: 37
- TAGS: bug,agents,backend,mcp

## Story

As the operator, I want the orchestrator's "who needs me" poll on a path that can
NEVER be parsed as an agent id, so it does not depend on route declaration order
and a stale build degrades to an honest 404 instead of a misleading "no such
agent".

## Context (the report)

BC3 added `GET /api/agents/pending`, a COLLECTION-level poll that lives under the
`/api/agents/{agent_id}` namespace. It is currently correct - declared before
`/api/agents/{agent_id}` (app.py), mirroring the `/api/agents/backends` guard, and
proven by `test_pending_agents_and_acknowledge_roundtrip` and a real server boot
(`GET /api/agents/pending -> [] 200`). BUT the ordering is load-bearing: a reorder
- or a stale dashboard build that predates the route - makes `/api/agents/{id}`
swallow "pending" and return `{"detail":"no such agent"}`, which is exactly the
confusing symptom an operator hit against a stale instance. Lesson
`static-route-before-param-route-or-it-is-shadowed` (20260723-094308) captured the
general trap; this task removes the dependence for this endpoint.

`POST /api/agents/{id}/acknowledge` is NOT affected - it has an `/acknowledge`
suffix, so `{id}` cannot swallow a sibling word.

## Steps

- [x] Move the poll to a collision-proof path: `GET /api/agents/pending` ->
      `GET /api/pending-agents` (a sibling of `/api/agents`, not a child, so route
      order is irrelevant). Handler + response model kept; docstring explains the
      collision-proof choice. Added `/api/pending-agents -> ["agents"]` to
      `_route_tags` (a new route needs its OpenAPI tag - the org test caught it).
- [x] Update the `pending_agents` MCP tool (`mcp_server.py`) to GET the new path;
      `acknowledge`'s docstring pointer updated too.
- [x] Update tests: the app round-trip + the mcp respx tool tests to the new path.
      The round-trip now also asserts the OLD `/api/agents/pending` is a plain 404
      (route vacated), and `test_openapi_docs_are_organized` asserts the new
      route's tag.
- [x] Update the (unreleased) CHANGELOG entry to the new path.

## Definition of Done

- `GET /api/pending-agents` returns the pending list; there is no
  `GET /api/agents/pending` route (the collision namespace is vacated).
  (test: `test_pending_agents_and_acknowledge_roundtrip`; cmd: real boot
  `curl /api/pending-agents -> []`)
- The `pending_agents` MCP tool calls `/api/pending-agents`.
  (test: `test_pending_agents_formats_the_poll`)
- `ruff check .`, `mypy`, `python -m pytest` green from the worktree.
  (cmd: `python -m pytest`)

## Notes

- Landed-code hardening of BC3 (`tasks/20260723-094308`); surfaced by an operator
  report during the BC-series flow.
- `acknowledge` unchanged (suffix path, no collision).

## Close record (2026-07-23)

What changed: `GET /api/agents/pending` -> `GET /api/pending-agents` (a sibling of
`/api/agents`, so it can never be parsed as `agent_id="pending"`); the
`pending_agents` MCP tool GETs the new path; `_route_tags` maps the new path to
the "agents" OpenAPI tag; CHANGELOG + docstrings updated. `acknowledge` untouched
(its `/acknowledge` suffix already prevents a collision).

Evidence: suite 367 passed; ruff + mypy clean. A REAL server boot of the worktree
source confirmed `GET /api/pending-agents -> [] 200` and the vacated
`GET /api/agents/pending -> 404`. Tests updated: the round-trip drives the new
path and asserts the old one 404s; `test_openapi_docs_are_organized` asserts the
new route's tag (it red-flagged the missing tag first - the value of a
"every route is tagged" invariant test).

Diagnosis (the operator report): the reported 404 `"no such agent"` was NOT a
code bug - the landed BC3 ordering was correct (a real boot of master served
`/api/agents/pending -> [] 200`). The operator's tool was reaching a build that
LACKED the route. Root cause surfaced while verifying: `scufris` runs the
uv2nix/nix-built package, so `nix develop --command scufris` (invoked at the main
checkout) serves the MAIN-CHECKOUT source, not a worktree's uncommitted edits -
and a process restart does not pick up landed code unless the build target
actually has it. Verifying a live route requires booting the intended source
(`cd <tree> && python -m scufris`, CWD-first on sys.path, like `python -m
pytest`), not `nix develop --command scufris` from elsewhere. Regardless, the
ordering dependence is now removed so a stale-build mismatch degrades to an
honest 404 rather than the misleading "no such agent".

Self-reflection: the first "definitive" boot I ran actually served master (the
nix package built from the main checkout), not the worktree - a reminder that
`nix develop --command <console-script>` is NOT the worktree's code. Use the
`python -m` CWD-first path to boot worktree source, the same reason the repo
mandates `python -m pytest` in a sprout.
</content>
