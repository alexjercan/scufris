# Review: Settings backend - config override store + gated writable endpoint

- TASK: 20260720-184136
- BRANCH: feature/settings-config-store

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (fresh subagent, ran the full nix suite; in-session
  pass re-ran the suite and adopted the three NIT fixes)

No BLOCKER/MAJOR findings. Full suite green (ruff + mypy + pytest via
`python -m pytest`). The whitelist is enforced at BOTH the API boundary
(`AgentConfigUpdate` extra=forbid -> 422) and the store (`UnknownSettingKey`),
each independently tested; a secret like `openai_api_key` cannot be written
through either path. The read-only gate is checked before any mutation/persist
(403, no file written). Rollback in `apply` is transactional. Corrupt/stale
state files are handled without crashing. In-place mutation with
`validate_assignment` does not leak across `create_app` instances. `AgentHandle`
implements the full protocol and carries the session across a rebuild.

- [x] R1.1 (MINOR) web/src/settings-view.ts:127 - stale "Read-only... restart
  to change" copy and the unused `writable` flag. This is the FRONTEND, owned
  by task 5; not in this backend task's scope.
  - Response: routed to task 5 - added an explicit "stale copy to fix" note in
    tasks/20260720-184148/TASK.md Notes (settings-view.ts:127 +
    webpack.config.js:56). Left unchanged here.
- [x] R1.2 (NIT) WRITABLE_KEYS vs AgentConfigUpdate fields are two hand-kept
  copies of the whitelist.
  - Response: fixed - added `test_writable_keys_match_the_api_update_model`
    asserting `set(AgentConfigUpdate.model_fields) == set(WRITABLE_KEYS)`, so a
    future key added to one is forced into the other.
- [x] R1.3 (NIT) settings_store `_persist` uses non-atomic `write_text`.
  - Response: fixed - write to a `.json.tmp` then `os.replace` for an atomic
    swap.
- [x] R1.4 (NIT) `AgentHandle.rebuild` drops the old inner without `aclose()`.
  - Response: fixed (documented) - added a comment noting every current
    backend's aclose is a no-op and rebuild is sync, so nothing leaks; a future
    connection-holding backend would need this made async.

No open `manual:` DoD items (all DoD proofs are `test:`/`cmd:`).
