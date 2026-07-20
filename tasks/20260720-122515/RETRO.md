# Retro: slash-command palette

- DATE: 20260720
- VERDICT: shipped

## What went well

- The spike had already answered the hard question (codex has no slash commands ->
  build it client-side), so this went straight to implementation with a clear
  shape: a `{name, description, run}` registry + a palette wired into the composer.
- The one real design conflict - Enter must accept a command when the palette is
  open but send when it is closed - was isolated in the keydown handler (check the
  palette first) and pinned with a "normal message still sends" test.
- Splitting the logic into pure exported helpers (`matchSlashCommands`,
  `chatMarkdown`) plus DOM interaction kept most of it unit-testable without
  driving the whole palette, and the interaction itself is testable via `initChat`.

## What went wrong / friction

- Nothing of substance. `/settings` (window.location.assign) and `/export`
  (Blob/URL) are not exercisable in jsdom, so those two commands lean on the
  guard-and-eyeball pattern; the fill/help/matching logic is fully tested.

## Lesson

- No new ledger entry. Reuses `side-effect-free-module-for-jsdom-tests` (pure
  helpers + thin wiring) and `frontend-verify-needs-e2e-serve` (browser-only APIs
  eyeballed / grepped from the built bundle).

## Follow-ups

- The registry is extensible: when the den MCP tools land (122514), `/today` etc.
  can call them directly instead of filling a prompt.
- Round-3 remaining: 122516 (attachments/previews), 122517 (settings console),
  122514 (den tools - awaiting the unified-CLI decision), 122518 (projects
  sub-spike), 122519 (nixos reconcile).
