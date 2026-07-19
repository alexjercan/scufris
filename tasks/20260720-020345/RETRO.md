# Retro: list app_server sessions (originator fix)

- DATE: 20260720
- VERDICT: fixed + verified on real data

## What happened

A one-line filter (`originator != "codex_exec"`) silently excluded every
app_server session because the two backends tag sessions with different
originators. The scary framing ("are they deleted?") was a filter bug, not data
loss - checking the disk first (the files were all there) reframed it instantly.

## Lesson

- `check-disk-before-assuming-data-loss` - when records "disappear" from a UI
  list, confirm the underlying files exist before touching anything; a missing
  list entry is far more often a filter/scope mismatch than a deletion.
- `backends-tag-provenance-differently` - `codex exec` and `codex app-server`
  write different `originator` values ("codex_exec" vs the app-server
  `clientInfo.name`). Any code that scopes by originator must accept the whole
  set scufris produces, or switching backends silently changes what is visible.

## Follow-ups

- If a third way of creating sessions appears, add its originator to
  `_SCUFRIS_ORIGINATORS` (or set a stable custom originator for both backends).
