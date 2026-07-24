# List app_server sessions in the switch list (originator fix)

- STATUS: CLOSED
- PRIORITY: 60
- TAGS: bug, agent, sessions

## Symptom

After switching the default backend to app_server, the session switch list was
empty on refresh - the user feared sessions were being deleted.

## Diagnosis

Nothing was deleted; the rollout files were all on disk. `list_sessions` scoped
the list to `originator == "codex_exec"`, but the app_server backend records
`originator = "scufris"` (codex takes it from the `clientInfo.name` we send on
`initialize`). The old `codex exec` path used codex's default "codex_exec". So
every app_server session was filtered out. Confirmed by reading the real
`~/.codex/sessions`: 9 rollouts for this cwd, all `originator: scufris`, zero
`codex_exec`.

## Fix

`scufris/sessions.py`: match a set of scufris-owned originators
(`_SCUFRIS_ORIGINATORS = {"codex_exec", "scufris"}`) instead of just
"codex_exec", so both backends' sessions list. Verified against the real
`~/.codex`: all 9 sessions now list.

## Tests

`test_list_sessions_filters_by_cwd_and_originator` extended: an
`originator="scufris"` (app_server) session lists alongside a "codex_exec" one,
while the "vscode" TUI session and an other-directory session stay excluded.

## Definition of Done

- [x] app_server sessions appear in the switch list.
- [x] exec ("codex_exec") sessions still appear.
- [x] Unrelated codex sessions (other originators / cwd) stay excluded.
- [x] Verified against real on-disk sessions; suite green.

## Closed: superseded by 20260724-111947

The narrow "fix the originator filter so app_server sessions show up" approach is
obsoleted by the session ownership index (tatr 20260724-111947, landed 236c129):
the switcher no longer infers ownership from a disk scan at all, so there is no
originator filter left to fix. See tasks/20260724-111947/DECISION.md.
