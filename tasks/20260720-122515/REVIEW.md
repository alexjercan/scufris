# Review: slash-command palette

- VERDICT: APPROVE
- ROUND: 1

## Summary

A client-side `/command` palette in the composer (codex has no slash commands).
An extensible `SLASH_COMMANDS` registry with `/new`, `/settings`, `/tasks`,
`/host`, `/export`, `/help`; a floating autocomplete that opens/filters on typing
`/`, is keyboard-driven (arrows, Enter/Tab accept, Esc dismiss) and mouse-clickable.
Frontend-only, no backend. 91 frontend tests green; built bundle ships the palette
and all six commands.

## What is good

- The keydown handler checks the palette FIRST, so Enter accepts the highlighted
  command instead of sending - the one real interaction conflict, handled cleanly;
  a normal message still sends when the palette is closed (pinned by a test).
- Pure exported helpers (`matchSlashCommands`, `chatMarkdown`) make the matching
  rule and the export format unit-testable without driving the whole palette; the
  interaction itself is also tested via `initChat` (open/filter/Enter/arrows/Esc).
- Sensible matching rule: only a lone `/token` at the very start with no space -
  once the user types an argument it is a real prompt, not a command. Guards
  whitespace (space and newline).
- Registry is extensible ({name, description, run}); commands split naturally into
  actions (`/new`, `/export`, `/help`), navigation (`/settings`), and prompt-fills
  (`/tasks`, `/host`) - the last are a deliberate two-step (fill then send) so the
  user reviews the prompt, matching the spike's "expand to a prompt template".
- Robust to environments: `/export` guards Blob/URL (absent in jsdom); the palette
  is a no-op when `#chat-palette` is missing (older DOM in other tests).

## Findings

- MINOR (accepted) - `/settings` uses `window.location.assign`, which jsdom does
  not implement, so it is not unit-tested (the fill/help/nav-less commands are).
  It works in the browser; the shipped markup + registry are grepped from the
  built bundle.
- MINOR (accepted) - the prompt-fill commands are a two-step (accept fills the
  composer, then the user sends). Intentional per the spike (review-before-send),
  but worth remembering if a "run immediately" command is ever wanted.
- NOTE - a11y is solid (role=listbox/option, aria-selected, full keyboard); a
  future `aria-activedescendant` would be the cherry on top, not needed now.

## Verdict

APPROVE. Clean, extensible, keyboard-first, and the Enter/submit conflict is
resolved and tested. Frontend-only with no backend surface. Visuals eyeballed per
`frontend-verify-needs-e2e-serve`.
