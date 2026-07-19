# Review: web/ TypeScript + webpack + Tailwind dashboard page

## Round 1 - 20260719

Scope: the new `web/` project (build config, `src/index.html`, `src/style.css`,
`src/main.ts`), `.gitignore`.

### Correctness / build

- `npm run ci` (prettier check + eslint recommendedTypeChecked + webpack build)
  is green; the strict tsconfig and typed-lint pass with no suppressions.
- End-to-end verified: after `npm run build`, the FastAPI backend located
  `web/dist`, served `/` (200) and `/main.js` (200), and `/api/stats` returned
  live data. The full slice runs.
- The `HostStats` TS interface mirrors the pydantic model exactly, so a field
  rename on either side surfaces at compile time on this side.
- Poll interval comes from `/api/config` with a sane fallback; fetch failures
  surface in the status bar and keep retrying rather than freezing the page.
- `node_modules/` and `web/dist/` are gitignored; `package-lock.json` is
  committed for reproducible installs.

### Observations (non-blocking)

- LOW (XSS surface): cards are built with `innerHTML` and interpolated values
  (hostname, disk mountpoints, os string). These come from the host's own psutil
  data on a single-user local dashboard, so the practical risk is minimal, but a
  mountpoint with an angle bracket would inject markup. A future hardening pass
  could switch to `textContent`/element construction or escape the values. Filed
  as a note, not a blocker for the first local slice.
- LOW (no DOM test): coverage is build + lint + the e2e serve check; there is no
  headless-DOM render test. Appropriate for a scaffold; a jsdom smoke test of
  `renderCards` is a reasonable follow-up if the render logic grows.
- NOTE: Tailwind v4 is wired via `@import "tailwindcss"` and used lightly; most
  styling is the custom stylesheet, matching the user's "custom style and CSS"
  ask.

### Verdict

APPROVE. The dashboard builds, passes typed lint, and is verified serving live
host stats end to end through the backend. The two LOW items (innerHTML escaping,
a DOM test) are sensible follow-ups, not blockers for the first running slice.
The user should eyeball the page in a browser to confirm the visual layout.
