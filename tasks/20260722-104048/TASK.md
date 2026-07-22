# broad styling pass over all pages - sharper terminal aesthetic from kitty config

- STATUS: IN_PROGRESS
- PRIORITY: 30
- TAGS: frontend, css, styling

## Goal

A broad styling pass over ALL pages (CSS + HTML): cleaner, sharper, more
consistent components - especially the buttons (the `/agents` page buttons are
called out as weak). Push the terminal aesthetic further. The user likes the
existing color scheme and the landing page, so keep the palette and refine, do
not redesign.

## Why

User feedback (2026-07-22): "the buttons (especially in /agents/) are not that
great, I would do a big review over all the pages + CSS + HTML and improve the
styling... cleaner and sharper, terminal style... check my kitty config + other
terminal related colors and style preferences to make it like that".

## Notes / scope to pin

- Read the user's kitty config (~/.config/kitty/ likely) and other terminal color
  prefs and align the web palette/typography to them where it improves the
  terminal feel. Keep the current scheme's character.
- Audit buttons/cards/forms/nav across every page for consistency; unify a button
  component; sharpen spacing/borders.
- Likely a /spike (inventory the current CSS + capture the kitty palette) then a
  staged implementation, page by page.
- LOW-PRIORITY sibling idea (ideation only, do NOT implement until much later):
  user-configurable styling/theming. Capture as a separate ideation task.
  (Captured as 20260722-104058.)

## Direction (pinned with the user 2026-07-22)

Kitty config read (~/.config/kitty/kitty.conf): warm-neutral, Iosevka, block
yellow cursor. Palette: bg #181818, fg #E4E4E4, red #F43841/#FF4F58, green
#73D936, yellow #FFDD33, blue #96A6C8, magenta #9E95C7, sage-cyan #95A99F, gray
#52494E. The user chose:

1. PALETTE: adopt the kitty palette (neutral #181818 bg / #E4E4E4 fg + kitty
   yellow/green/red/sage/gray), replacing the current cool blue-black + its teal
   scheme's NEUTRALS.
2. ACCENT: KEEP the app's cyan #47d4e0 as the PRIMARY accent; use yellow #FFDD33
   for focus/active (cursor-like). (So cyan stays; only neutrals + semantic
   colors move to kitty.)
3. TYPOGRAPHY: full monospace everywhere (Iosevka-first stack, fall back to
   JetBrains Mono / system mono; no fonts are bundled). `--font-body` -> the mono
   stack (only 4 sites use it).
4. EDGES: sharpen radius to ~2px and DROP the body radial glow.

Concrete bug found: `.settings__btn` sets `background: var(--bg)` but the palette
defines only `--bg-0`/`--bg-1` - `--bg` is UNDEFINED, so every button renders with
no fill (just a thin border). That is why the buttons read as weak. Fix as part of
the button pass.

## Steps (/plan)

- [ ] `:root` tokens (style.css): adopt the kitty neutrals + semantics - `--bg-0`
      #181818, `--bg-1`/`--panel`/`--panel-2` as stepped neutral grays, `--border`
      a kitty-gray (#52494E-ish), `--text` #E4E4E4, `--text-muted` the sage
      #95A99F; `--green` #73D936, `--red` #F43841 (+ a bright #FF4F58 for hover),
      `--amber` -> yellow #FFDD33; ADD `--yellow`/`--focus` #FFDD33; KEEP `--cyan`
      #47d4e0. `--radius` 2px. Point `--font-body` at the mono stack (Iosevka
      first). Remove the `body` radial-gradient glow.
- [ ] Buttons: give `.settings__btn` a real fill (a defined token, not the broken
      `var(--bg)`), a consistent hover (raise bg + border to cyan) and a yellow
      focus-visible ring; keep the danger variant. This is THE shared button - make
      the /agents (and everywhere) buttons read as solid, tactile controls.
- [ ] Component consistency sweep across the shared stylesheet: nav active state
      (cyan text + a yellow underline/marker), cards/panels (flatter, 2px, kitty
      border), inputs/selects (mono, 2px, focus ring), badges/tags - all on the
      tokens so every page inherits the terminal look. No per-page color literals.
- [ ] A CSS token-integrity test (new `web/src/style-tokens.test.ts`): parse
      `style.css`, collect the `:root` custom-property definitions and every
      `var(--x)` reference WITHOUT a fallback, and assert the referenced set is a
      subset of the defined set - pinning the `var(--bg)`-undefined class of bug so
      it cannot regress. (Ties into the pending `render-rewrite-orphans-its-css`
      promotion: styled-but-undefined tokens.)
- [ ] Verify: the full web suite stays green (no structural regressions), `npm run
      ci` (format/lint/test/build) green.

## Definition of Done

- The web palette adopts the kitty neutrals (#181818 bg, #E4E4E4 fg) + kitty
  yellow/green/red/sage, with cyan #47d4e0 kept as the primary accent and yellow
  #FFDD33 for focus; typography is full monospace; radius is ~2px and the radial
  glow is gone. (test: the token-integrity test + `npm run build`; manual: the
  pages read like the terminal.)
- Buttons have a real fill + consistent hover/focus across every page, and the
  `var(--bg)`-undefined bug is gone. (test: token-integrity asserts no undefined
  `var(--x)`; manual: the /agents buttons look solid, not ghost outlines.)
- No structural regressions: the full web suite + build are green.
- manual: eyeball `/`, `/agents`, `/projects`, `/settings`, `/stats`, and the
  agent + project detail pages - consistent, sharper, terminal-like, palette
  intact.
