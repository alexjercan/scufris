# Review: Settings UI - interactive config controls + tools editing

- TASK: 20260720-184148
- BRANCH: feature/settings-ui-controls

## Round 1

- VERDICT: REQUEST_CHANGES
- REVIEWER: out-of-context (fresh subagent; recreated the node_modules symlink,
  ran full `npm run ci` + backend suite, removed the symlink after)

Frontend (107 tests) and backend green. Reviewer verified XSS escaping on the
writable server list and tool cards (incl. a hostile-id test), single
authoritative render (reload-after-mutation, no client copy), full
`disabled_tools` set rebuilt on a tool toggle, confirm-and-revert on high-impact
turn-OFF and server removal, server-side list rebuild from `settings.mcp_servers`,
and removal of the stale "restart to change" copy.

- [x] R1.1 (MAJOR) scufris/app.py - the net-new `POST /api/agent/mcp_servers`
  and `DELETE /api/agent/mcp_servers/{id}` endpoints had NO direct tests; only
  the pre-existing PATCH path was covered, so the unique branches (409 dup, 404
  missing, 403 read-only gate, 422 bad/reserved id, incremental append/remove)
  were unverified.
  - Response: fixed - added `test_post_mcp_server_appends_and_persists`,
    `test_post_mcp_server_rejects_duplicate` (409),
    `test_post_mcp_server_rejects_bad_or_reserved_id` (422, parametrized),
    `test_delete_mcp_server_removes_and_404s_unknown` (200 + 404), and
    `test_mcp_server_endpoints_forbidden_when_readonly` (403 on both verbs). 12
    mcp_server tests pass; each asserts the branch's own status/effect.
- [x] R1.2 (NIT) app.py `remove_mcp_server` takes `server_id` from the path with
  no validation, falling through to 404 for garbage.
  - Response: left as-is by design - the 404 is correct and nothing is persisted
    for an unknown/garbage id; validating would only change the status of a
    non-existent target, not prevent anything.

## Round 2

- VERDICT: APPROVE
- REVIEWER: in-session (the round-1 finding was "add the missing endpoint
  tests"; the added tests are the exact regressions requested and each fails if
  its branch regresses - e.g. dropping the dup check makes the 409 test 200).
  Re-ran the full suites: `npm run ci` 107 tests green, backend `python -m
  pytest` 189 passed, ruff + mypy clean.

No open `manual:` DoD items beyond the ones now covered by the added tests.
