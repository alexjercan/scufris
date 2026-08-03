# Notes: Clear the round-1 MINOR findings from the diagnostics alignment

## What changes

One operator-visible fix and five hygiene items. Before: `/settings` on Telegram
prints the fallback quota sentence for an account whose backend reports only a
`secondary` rate window, while `/settings usage` prints that window - the two
disagree. After: both read the same windows. The other five findings move
pointers, comments and exports into line with the code; no behavior.

## Surfaces

| File | Why |
|-|-|
| `scufris/telegram/render.py:388` (`render_settings_summary`) | R1.2, the one real bug: `usage.primary` only |
| `scufris/telegram/render.py:319-345` (`render_usage`) | R1.4, the 12-line `windows` comprehension to simplify |
| `tests/test_telegram.py` | extend `test_render_settings_summary_carries_the_capability_reading` |
| `scufris/README.md:438` | R1.1, wrong module for `capabilityText` (finding says :356 - stale, see below) |
| `scufris/telegram/text.py:58` | R1.1, same wrong module in a comment |
| `web/src/agent-settings-view.ts:62` | R1.1, same |
| `CHANGELOG.md` | R1.3, no bullet for the three-state operator wording |
| `web/src/agent-view.ts:151` | R1.5, a task ID in a code comment (AGENTS.md forbids) |
| `web/src/agent-settings-panels.ts:45,140` | R1.6, two unimported exports (see below) |
| `web/src/chat-sidebar.ts:96` + `web/src/agent-settings-panels.ts:100` | separable retro item: `resetsIn` duplicated verbatim |

## Data and interfaces

Nothing structural. Signatures that move or narrow:

- `render_settings_summary(info, tools, health) -> str` - unchanged signature,
  the quota read inside it widens to `primary or secondary`.
- `resetsIn(resetsAt: number | null): string` - if deduped, it moves to
  `web/src/common.ts` and both current sites import it. Both copies are
  byte-identical today.
- `capabilityPanel` and `capabilityText` in `agent-settings-panels.ts` lose
  `export` (they stay module-local; `capabilityText` is called at lines 158 and
  184 of its own file).

## Sketches

R1.2, illustrative:

```python
    usage = info.quota.value
-   primary = usage.primary if usage is not None else None
+   window = None if usage is None else (usage.primary or usage.secondary)
    usage_line = (
-       f"{primary.used_percent:.0f}% ({_fmt_window(primary.window_minutes)})"
-       if primary is not None
+       f"{window.used_percent:.0f}% ({_fmt_window(window.window_minutes)})"
+       if window is not None
        else _quota_reading(info)
    )
```

R1.4, illustrative - the comprehension plus the redundant `usage is not None`
guard at line 340 collapse back toward the 3-line guard they replaced:

```python
-   windows = ([] if usage is None else
-              [(label, w) for label, w in (("primary", usage.primary),
-                                           ("secondary", usage.secondary)) if w is not None])
-   if not windows: return _fenced("Usage", _quota_reading(info))
-   if usage is not None and usage.plan_type: ...
+   if usage is None or (usage.primary is None and usage.secondary is None):
+       return _fenced("Usage", _quota_reading(info))
+   if usage.plan_type: ...
```

## Shape

```
info.quota : Capability[UsageQuota]
                 |
   +-------------+-------------+
   |                           |
render_usage (/settings usage) render_settings_summary (/settings)
  primary AND secondary          primary ONLY   <-- R1.2: disagree when
                                                    primary is None
```

## Consequences and open questions

- **Three of the six pointers in TASK.md have drifted** since it was written.
  Re-derive at plan time; do not trust the line numbers:
  - R1.1: `capabilityText` in `scufris/README.md` is at **line 438**, not 356.
  - R1.6: of the three symbols, only **`capabilityPanel` (now line 45)** and
    **`capabilityText` (line 140)** are unimported. **`statusPanel` (line 114,
    the finding says 116) IS imported and used** at
    `agent-settings-view.ts:32,215` - dropping its export would break the build.
    The finding is 2/3 right.
  - `resetsIn` is at `agent-settings-panels.ts:100` (the finding says 116).
- R1.2 is the only item with a user-visible effect and the only one that needs a
  proof; the rest are covered by the existing suites plus `rg` greps.
- The `resetsIn` dedupe is pre-existing and separable. Recommend keeping it in
  scope here - it is ten lines and touches the same two files R1.6 does - but
  splitting it out is defensible if the diff wants to stay pure-cleanup.
- Open question: does R1.3's CHANGELOG bullet go under the existing
  `[Unreleased] ### Changed` block (which already describes the capability
  envelope) or as its own bullet for the operator wording? Assumption taken: a
  separate bullet under the same heading, because the wording is what the
  operator sees and the envelope bullet is about the API shape.
