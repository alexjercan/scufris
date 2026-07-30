# Retro: Add the dashboard host approval queue and audit surface

- TASK: 20260730-104520
- BRANCH: feat/host-approval-queue
- REVIEW ROUNDS: 2 (5 findings: 2 MAJOR, 2 MINOR, 1 NIT)

What shipped and why is in `NOTES.md`. This is process only.

## What went well

- **Serving the page for real, before trusting the shapes.** Curling the two
  endpoints the page reads took a minute and found the one defect no test could:
  `/api/host/actions` answers `200 []` when the helper is absent, so the
  "not configured" branch keyed on it would never have fired. 247 vitest tests, the
  whole python suite and the webpack build were green with that bug in.
- **Probing the view instead of reading it.** Both MAJOR findings came from driving
  the built module in jsdom and printing what happened - a typed token vanishing on
  the next poll, an error banner that never cleared. Neither was visible by reading
  the code with the intent still fresh in mind; both were obvious the moment the
  probe printed `""`.
- **Sabotage-testing the escaping test.** Switching `text()` to `innerHTML` turned
  exactly one test red, which is the evidence that the XSS pin can fail. A pin that
  cannot fail is the thing this repo's ledger already paid for once.
- **The repo's own guards did their job.** The API-seam test caught the EventSource,
  the strict tsc build caught a closure-narrowing error vitest tolerated, and the
  pre-commit hook caught the staged `node_modules` symlink. Three classes of mistake
  absorbed by tooling rather than by care.

## What went wrong

- **The two MAJORs are the same root cause: I designed the render and never
  simulated a SESSION.** `replaceChildren` on a 4-second poll is obviously fine for
  a read-only dashboard (which is what every other page here is) and obviously wrong
  for a page with inputs the operator must type into - and the one-way gate, the
  feature most of the design effort went into, is the one it broke. Same for the
  error banner: the code was written for the moment a decision fails, not for the
  minute afterwards. Root cause: I tested each render in isolation and never asked
  what the page does over TIME, with a person interacting with it.
- **The plan step named a mechanism that does not exist** ("the FastAPI page
  route"). Two minutes reading how `/stats/` is served at plan time would have
  removed it; instead it survived into the step list and had to be corrected while
  ticking it.
- **The first escaping assertion was wrong, not the code.** `innerHTML` legitimately
  contains the raw string when a value is set via the `title` DOM property, which is
  never parsed as markup. Grepping serialised HTML for `<img` conflates "was parsed"
  with "appears somewhere".

## What to improve next time

- For any page with an input, write the SESSION test before the render is finished:
  type, let a poll land, and assert the value and the focus survived. It is three
  lines and it is the difference between a usable control and a race.
- Transient UI state (an error banner, a spinner, a toast) needs its clearing path
  written in the same edit as its setting path. "Who clears this, and when?" is the
  same question that caught the BLOCKED deadlock in the previous task - a state with
  no clearer is the recurring shape here.
- Assert "did this data become markup" structurally (element count against a clean
  render), never by grepping serialised HTML.

## Action items

- [x] Lessons ledger: `poll-render-wipes-the-input-under-the-operator` and
      `transient-ui-state-needs-its-clearing-path-in-the-same-edit` appended;
      `frontend-verify-needs-e2e-serve` bumped to x2 with this task's evidence.
- [x] The `manual:` DoD item is carried to the epic's Manual Acceptance list with
      the note that no browser tooling existed in this session, so the visual render
      is unverified.
- [ ] Not created as a task: an out-of-context review round, and a real
      browser/phone look at the page - both need mechanisms this session lacks.
