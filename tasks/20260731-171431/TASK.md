# Split the oversized frontend views under the size cap

- STATUS: OPEN
- PRIORITY: 75
- TAGS: refactor, v0.2.0, frontend, maintainability
- KIND: TASK
- FLOW STEP: PLANNED
- PLAN STATUS: APPROVED
- PARENT: 20260731-171411
- DEPENDS ON: 20260731-171420

## Story

As a maintainer, I want the four oversized frontend modules split by view
concern, so that a page change does not load a thousand lines of unrelated
rendering and fetch code.

## Steps

- [ ] Record the pre-move baseline before touching anything: `npm run ci` green
      (22 test files, 258 tests), `python scripts/check_file_size.py` exit 0,
      `npx prettier --check src/<file>` and `npx eslint src/<file>` clean on all
      four subjects. Any later flag is then provably introduced, not inherited.
      Read a base file with `git show master:<path>`, NEVER `git checkout
      master`.
- [ ] Write two throwaway checks under the scratchpad and use them on every
      split below: (a) a surface diff that lists the top-level `export` names of
      `git show master:web/src/<file>.ts` and differences them against the union
      of exports across the new sibling set, so a dropped public name is
      reported rather than noticed; (b) a move proof that normalizes both sides
      to a multiset of stripped, non-blank, non-comment, non-import lines and
      differences them, so every remaining entry is a rewording or a rename that
      can be named. Grep-based export extraction is sound here: `rg '^export \{'
      web/src` finds no re-export blocks, so every export is a top-level
      declaration.
- [ ] Commit 1 - `web/src/common.ts` (834). Extract `stats-types.ts` (the
      `/api/stats` + `/api/host/overview` wire shapes), `host-types.ts` (the
      `/api/host/actions` + `/api/host/audit` wire shapes) and `agent-types.ts`
      (chat, agent, session, usage, project). `common.ts` keeps the runtime:
      `AppConfig`/`loadConfig`, `el`, `escapeHtml`, `formatBytes`, the CSRF
      constants, `csrfToken`, `goToLogin`, `apiFetch`, `logout`, `fetchJson`,
      `sendJson`, and the shared display labels. Repoint the ~31 importers,
      including the `.test.ts` files. `git add` the new files, delete the
      `web/src/common.ts` ALLOWLIST entry in the same commit.
- [ ] Commit 2 - `web/src/agent-chat-view.ts` (1106). Extract
      `agent-chat-types.ts`, `agent-chat-log.ts`, `agent-chat-turn.ts`
      (`createTurnRunner`, owning `streaming` and `cancelCurrent`, taking
      explicit deps and `appendMessage`/`lastMessage` callbacks rather than the
      reassigned `msgs` array) and `agent-chat-composer.ts` (`createSlashPalette`,
      `createImageAttach`, `autosize`). Repoint `agent-view.ts`,
      `agent-detail.ts` and `agent-chat-view.test.ts`. Delete the entry.
- [ ] Commit 3 - `web/src/host-view.ts` (1022). Extract `host-format.ts`,
      `host-actions.ts` (the `HostActions` contract, `dispatch`, and the
      `lastError` state it writes), `host-proposal.ts`, `host-history.ts` and
      `host-checks.ts`. Repoint `host.ts` and `host-view.test.ts`. Delete the
      entry.
- [ ] Commit 4 - `web/src/stats-view.ts` (870). Extract `stats-elements.ts`,
      `stats-cards.ts` and `stats-host-cards.ts`. Repoint `stats.ts` and
      `stats-view.test.ts`. Delete the entry, and update `web/README.md`'s
      "View logic is separated from the DOM entry" convention plus its file list
      to the new layout in this commit.
- [ ] Apply the `AGENTS.md` comment policy to every comment that moves: drop the
      review-round citations in `host-view.ts` and the ledger-key citations in
      `host-view.ts` and `stats-view.ts`, keeping each invariant as a fact about
      the code; introduce no task ID. Confirm the exact set with a grep rather
      than from the DECISION.md list.
- [ ] After each commit, before `nix build`: `git add` every new file (untracked
      files are invisible to nix), then run `python scripts/check_file_size.py`,
      `npm run ci`, and check no touched file crossed the cap -
      `agent-settings-view.ts` is 589 and gains import lines.
- [ ] Prove the guard at every commit, not only the tip:
      `git rebase master --exec 'python scripts/check_file_size.py'`.

## Definition of Done

- No non-test file under `web/src/` exceeds 600 lines, and the ALLOWLIST no
  longer names `web/src/common.ts`, `web/src/agent-chat-view.ts`,
  `web/src/host-view.ts` or `web/src/stats-view.ts`
  (cmd: `python scripts/check_file_size.py && rg -n "web/src" scripts/check_file_size.py`).
- `web/src/agent-chat-view.test.ts` and `web/src/host-view.test.ts` keep their
  ALLOWLIST entries, hold no fewer `it(` cases than on `master`, and contain no
  moved assertions - 20260731-171432 owns splitting them
  (cmd: `rg -c "^\s+it\b" web/src/agent-chat-view.test.ts web/src/host-view.test.ts`).
- Every top-level export of each base file still exists somewhere in its new
  sibling set, or the DECISION/NOTES record names the drop and why
  (cmd: the surface diff from Steps, base vs branch).
- The split is a move: the normalized code-line multiset difference between each
  base file and its new sibling set is empty except for named rewordings
  (cmd: the move proof from Steps).
- Frontend gate passes (cmd: `cd web && npm run ci`).
- Package build passes (cmd: `nix build .#scufris-web`).
- The size guard is green at every commit, not only the tip
  (cmd: `git rebase master --exec 'python scripts/check_file_size.py'`).
- Authenticated calls still route through one `apiFetch`, defined once in
  `web/src/common.ts`
  (cmd: `rg -n "export async function apiFetch" web/src`).
- No comment in `web/src` cites a task, review round or ledger key as its only
  justification
  (cmd: `rg -n "2026[0-9]{4}-[0-9]{6}|review round|see the lesson|the ledger" web/src`).
- `web/README.md` conventions and file list match the new layout
  (cmd: `rg -n "view" web/README.md`).

## Notes

- Epic: 20260731-171411. Depends on: 20260731-171420.
- Load-bearing choices in DECISION.md: no facade/barrel (the Python shape does
  NOT translate - `tsc` makes a missed repoint a build failure, and the
  frontend has no string-addressed module targets); `common.ts` splits by
  concern into three sibling type modules and does not become a directory
  (a single `types.ts` measures ~653 lines against a 600 cap); four commits, one
  per over-cap file, `common.ts` first.
- SCOPE BOUNDARY, settled: this task deletes exactly the four source ALLOWLIST
  entries above. `web/src/agent-chat-view.test.ts` (1183) and
  `web/src/host-view.test.ts` (997) stay allowlisted and belong to
  20260731-171432. Test files change IMPORT LINES ONLY here; no assertion,
  fixture or `describe` block moves, and no new `.test.ts` file is created.
  Moving test code out would risk dropping either file under 900, which the
  guard fails as a stale entry. This supersedes the earlier "test files move
  with their subject" note.
- `apiFetch` and the auth bootstrap contract stay in `web/src/common.ts`,
  unduplicated, as `AGENTS.md` pins.
- Baseline recorded at plan time on `8bfbe74`: `npm run ci` green, 22 test
  files / 258 tests, `check_file_size.py` exit 0.
- No behavior change and no new abstraction a single caller does not demand.
  A behavior fix found on the way becomes its own task.
- `nix flake check` and `nix build` evaluate only git-TRACKED files: `git add`
  new files before running either.
- Commit the `tatr flow` transition, or re-run it, before any `git reset --hard`
  or rebase - flow state lives in this working-tree file.
