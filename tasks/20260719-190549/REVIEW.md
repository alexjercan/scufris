# Review: Header/footer as shared fragments + polish

## Round 1 - 20260719

Scope: `web/webpack-partials.js` (new plugin), `web/src/_header.html`,
`web/src/_footer.html`, `web/src/index.html` + `stats.html` (placeholders),
`web/webpack.config.js`, `web/src/style.css` (host-summary), `web/package.json`.

### Correctness

- Single-source achieved and verified: both `/` and `/stats/` render the brand +
  2 nav links + status footer, and the `<div id="header"></div>` placeholder is
  gone from the output (`leftover=0`) - so the markup is injected from
  `_header.html`/`_footer.html`, not duplicated in the two page templates. The
  `<%= basePath %>` links resolve (`href="/"`, `href="/stats/"`).
- The plugin faithfully mirrors nova-protocol's `HtmlPartialsPlugin` (beforeEmit
  hook, read partial, substitute basePath, replace placeholder), trimmed to
  header+footer. Reusable via the `basePath` option if a deploy subpath is ever
  needed.
- Nav active-highlighting still works: `nav.ts` `initNav` runs at runtime over
  the `.nav__link`s, which are now in the injected DOM - unchanged logic, and the
  links are present on both pages.
- Polish: `.host-summary` gains a top margin (off the delimiter line) and each
  item a padding + left divider (first item none), directly addressing the
  "nixosos" run-together the user reported.
- `npm run ci` green (prettier now covers `webpack-partials.js`; 11 jsdom tests
  unchanged); python checks unaffected.

### Observations (non-blocking)

- MINOR: the partial injection is verified by the serve smoke (curling both built
  pages), not a unit test - a build-output assertion would mean building inside a
  test, which is heavier than the check is worth here. The jsdom render tests are
  unchanged and still green.
- MINOR: the shared footer's `#status` starts empty; on the agent page it stays
  empty (that page does not poll stats / call setStatus), exactly as before.
- NIT: `basePath` is `/` (single local host); fine, and future-proofed by the
  plugin option.

### Verdict

- VERDICT: APPROVE

Meets the Definition of Done: the header and footer are single-source
partials injected at build time (no duplicated markup), both built pages carry
them, the nav still highlights the current page, and the host-summary is spaced +
divided so its fields no longer run together. Checks green; serve-verified.
