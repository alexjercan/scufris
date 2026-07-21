# Retro: F6 per-backend model autocomplete

- TASK: 20260721-152737
- BRANCH: feature/model-dropdown
- REVIEW ROUNDS: 1 (out-of-context APPROVE, zero findings)

## What went well

- The MB1 groundwork (server-authoritative `GET /api/agents/backends`) made this
  a small additive change: one field (`models`) on the existing payload + one
  helper (`models_for`), then the frontend datalist. No new endpoint.
- Chose the datalist over a hard `<select>` deliberately, so the model field
  stays a dropdown of suggestions AND keeps the free-text escape hatch (custom
  model ids still work) - covered by a test.
- `models_for` prepending the configured default (when outside the catalog) is
  the subtle correctness bit: an env-overridden `claude_model` would otherwise be
  absent from its own picker. Pinned by a dedicated env-override endpoint test.
- Zero-finding review - the change was small and every edge (id collision,
  free-text, prepend) was tested up front.

## What went wrong

- Nothing broke. One minor friction: the `<datalist>` must be in the DOM for the
  input's `list` reference to resolve, but callers append `fields.model`
  directly. Solved by exposing `fields.modelList` and having both callers append
  it next to the input - a small API wart but explicit and testable.

## What to improve next time

- When a form control needs an associated hidden element (datalist, a label
  bound by `for`), expose it from the builder as a distinct field and document
  that callers must append it - do not rely on it being auto-inserted.

## Action items

- [x] Review APPROVE, no follow-ups.
- Next: the exec-drop + docs task (drop the codex exec MODE, app_server-only;
  refresh .env.example + README for Agents v2).
