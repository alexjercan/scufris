# Header/footer as shared fragments (nova-style) + polish

- STATUS: OPEN
- PRIORITY: 8
- TAGS: feature,backlog,dashboard,ui

## Goal

Extract the page header (brand + nav) and footer (status bar) into shared HTML
fragments/partials injected at build time - like nova-protocol's HtmlPartialsPlugin
- so they are single-source across pages, and improve their visual polish.

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
