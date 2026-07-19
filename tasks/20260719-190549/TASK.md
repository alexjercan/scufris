# Header/footer as shared fragments (nova-style) + polish

- STATUS: CLOSED
- PRIORITY: 8
- TAGS: feature, backlog, dashboard, ui

## Goal

Extract the page header (brand + nav) and footer (status bar) into shared HTML
fragments/partials injected at build time - like nova-protocol's HtmlPartialsPlugin
- so they are single-source across pages, and improve their visual polish.

## Steps

- [x] Add `web/webpack-partials.js` - an `HtmlPartialsPlugin` (mirroring
      nova-protocol/web/webpack-partials.js): on HtmlWebpackPlugin's `beforeEmit`,
      read `src/_header.html` / `src/_footer.html`, replace `<%= basePath %>` with
      the base path (default `/`), and inject into `<div id="header"></div>` /
      `<div id="footer"></div>` placeholders.
- [x] Create `web/src/_header.html` (the topbar: brand + Agent|Stats nav, links
      via `<%= basePath %>` and `<%= basePath %>stats/`) and `web/src/_footer.html`
      (the statusbar with `#status`).
- [x] Replace the duplicated inline header/footer in `web/src/index.html` and
      `web/src/stats.html` with the two placeholders; keep the page-specific
      middle (chat on index, host-summary + cards on stats).
- [x] Wire `HtmlPartialsPlugin` into `webpack.config.js`; add `webpack-partials.js`
      to the prettier format globs.
- [x] Polish CSS: give `.host-summary` a top margin so it is not glued to the
      topbar delimiter, and separate its items (wider gap + a subtle divider) so
      "host nixos" / "os Linux" no longer read as "nixosos Linux".
- [x] LIVE serve smoke: `/` and `/stats/` both render the brand + both nav links
      + the status footer FROM THE SHARED PARTIALS (grep the built pages); nav
      active link still highlights (initNav). `ruff`/`mypy`/`pytest` + `npm run ci`
      green.

## Definition of Done

- The header and footer come from single-source `_header.html` / `_footer.html`
  partials injected at build time; `index.html` and `stats.html` no longer
  duplicate that markup.
- Both built pages contain the header + footer; nav still highlights the current
  page; the host-summary is spaced from the delimiter and its items are visually
  separated. `npm run ci` + python checks green; serve-verified.

## Feedback captured (user, 2026-07-19)

- Make the header/footer FRAGMENTS like nova-protocol does: `web/src/_header.html`
  + `web/src/_footer.html` injected into `<div id="header">` / `<div id="footer">`
  placeholders by a small webpack partials plugin (nova has
  `web/webpack-partials.js` + `_header.html`/`_footer.html`). Today each page
  (`index.html`, `stats.html`) duplicates the header/nav + footer markup.
- Polish the look:
  - The host-summary line (`host nixos  os Linux 6.18.37  up 9h 51m`) sits too
    close to the topbar delimiter line - add spacing/margin below the border.
  - The summary items visually run together ("host nixos" + "os Linux" reads as
    "nixosos Linux") - widen the gap or add separators between items.
  - General header/footer visual improvements welcome; the overall style is good.

## Notes

- Reference the nova-protocol pattern at ~/personal/nova-protocol/web
  (`webpack-partials.js`, `src/_header.html`, `src/_footer.html`, and the
  `<div id="header"></div>` / `<div id="footer">` placeholders + the `initSite`
  nav-active logic). Scufris already has `web/src/nav.ts` (`initNav`) for the
  active link - keep or fold into the shared header.
- Current header/footer live inline in `web/src/index.html` and
  `web/src/stats.html`; the multi-page build is two entries + one
  HtmlWebpackPlugin per page (tatr 20260719-180543).
- Lower priority - user said the header "can get improvements later on."

## Implementation

- `web/webpack-partials.js`: an `HtmlPartialsPlugin` mirroring nova-protocol's -
  on HtmlWebpackPlugin `beforeEmit`, reads `src/_header.html`/`src/_footer.html`,
  substitutes `<%= basePath %>` (default `/`), and injects into the
  `<div id="header"></div>` / `<div id="footer"></div>` placeholders. Wired into
  `webpack.config.js` with `basePath: "/"`.
- `web/src/_header.html` (topbar: brand + Agent|Stats nav via `<%= basePath %>`)
  and `web/src/_footer.html` (statusbar `#status`) are the single source.
  `index.html` and `stats.html` now hold only the placeholders + their
  page-specific middle (chat / host-summary + cards). No duplicated header/footer.
- Polish: `.host-summary` gets `margin-top` (off the topbar delimiter) and each
  item a padding + left divider (first item none), so "host nixos" / "os Linux"
  read as separate fields, not "nixosos".
- `webpack-partials.js` added to the prettier globs.
- Verified: `npm run ci` green (11 jsdom tests unchanged); serve smoke - both `/`
  and `/stats/` carry the brand + 2 nav links + status footer from the partials
  (placeholders replaced), links resolve via basePath.
