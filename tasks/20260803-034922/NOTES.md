# Notes: Pin the two legacy-diagnostics tests that cannot go red

## What changes

No product behavior changes. Two tests that pass whether or not their guarded
change exists are made falsifiable, so the two claims they carry - DECISION-4
("a disabled agent is supported, not unsupported") and the frontend's capability
unwrap - are actually held by the suite.

Before: revert either change and the suite stays green. After: reverting either
turns its test red.

## Surfaces

| File | Why |
|-|-|
| `tests/test_app.py:1858` (`test_disabled_agent_is_supported_not_unsupported`) | uses an empty `codex_home`, so the deleted short-circuit and the delegated reader give the same answer |
| `web/src/agent-view.test.ts:386` ("hides the meter when the backend cannot report usage") | the raw envelope and the unwrapped value both hide the meter |
| `web/src/agent-view.ts:145-158` (`loadUsage`) | the unwrap under test; read only |
| `web/src/chat-sidebar.ts:161-170` (`renderUsage`) | reads `usage?.primary`; explains why the case cannot discriminate |

## Data and interfaces

Nothing added or changed. The two levers already exist:

- `_write_session_rollout(home, id, cwd=os.getcwd())` (used at
  `tests/test_app.py:1846`) populates a codex home.
- `stubUsageFetch(envelope)` (`web/src/agent-view.test.ts`) drives the fetch.

## Sketches

R2.1 - populate the home so the two readings differ (illustrative):

```python
def test_disabled_agent_is_supported_not_unsupported(...):
    home = tmp_path / "codex"
-   # codex_home=tmp_path / "no-codex"  (empty: nothing to read either way)
+   _write_session_rollout(home, "sess-d", cwd=os.getcwd(), used_percent=42.0)
    app = create_app(settings=Settings(..., codex_home=home, agent_enabled=False))
    ...
-   assert memory["value"]["session_count"] == 0
+   assert memory["value"]["session_count"] == 1        # the REAL reading
+   assert body_usage["value"]["primary"]["used_percent"] == 42.0
    assert account["enabled"] is False                  # only this carries disabled
```

Red-first check: restore the short-circuit from `git show master:scufris/app.py`
(pre-delegation), rerun, expect red on the two new asserts.

## Shape

```
R2.1  agent_enabled=False + POPULATED codex_home
        old code:  short-circuit  -> usage/memory = empty      \  differ ->
        new code:  delegate       -> usage/memory = real       /  test can go red
      (with an EMPTY home both paths return empty -> the test proves nothing)

R2.2  loadUsage -> renderUsage(x)
        unwrapped:  x = quota.value      -> x is null          -> hidden
        raw:        x = quota (envelope) -> x.primary undefined -> hidden
      supported:false discriminates NOTHING. supported:true + a primary window
      is the only shape where the two paths differ - and that case already exists
      directly above at agent-view.test.ts:375.
```

## Consequences and open questions

- **R2.2's first suggested fix does not work.** "Assert the meter is empty as
  well as hidden" does not discriminate: `renderUsage` calls
  `meter.replaceChildren()` on both null and a primary-less object, so the DOM
  is identical. The envelope only differs from the value when
  `supported: true` and a `primary` window is present - which is the "shows the
  meter" case at `agent-view.test.ts:375`. **Recommendation: take R2.2's second
  option and delete the case as covered**, or, if a negative case is wanted,
  restate it as `{supported: true, value: {primary: null, secondary: {...}}}`,
  which the raw envelope would also mis-handle. Decide this in planning; it is
  the one place the two readings of the finding produce different work.
- R2.1 is unambiguous and cheap: one populated home plus two asserts.
- Cost: `test_disabled_agent_is_supported_not_unsupported` becomes coupled to
  `_write_session_rollout`'s fixture shape. Acceptable - three neighbouring
  tests already are.
- The DoD proof (`python -m pytest && cd web && npm run ci`) only shows green.
  The red half must be demonstrated by hand and recorded in the task record;
  that is the whole point of the task and should be an explicit Step, not a
  proof command.
