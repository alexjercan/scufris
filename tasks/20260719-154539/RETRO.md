# Retro: web/ TypeScript + webpack + Tailwind dashboard page

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- Mirroring the nova-protocol `web/` scaffolding and trimming to a single entry
  point made the build config near-copy-paste; webpack + ts-loader + Tailwind v4
  built first try, no toolchain yak-shaving.
- Typing the `/api/stats` payload as a `HostStats` interface that mirrors the
  pydantic model gives a compile-time contract between backend and frontend.
- The end-to-end serve check (backend finds `web/dist`, serves `/` + `/main.js`,
  `/api/stats` returns live data) proved the whole slice wired together, which a
  build-only check would have missed.
- `nodejs` was already in the dev shell from the backend task, so `npm` just
  worked under `nix develop` - the dependency ordering paid off.

## What went wrong / friction

- eslint `recommendedTypeChecked` is strict about `any`: `resp.json()` is `any`,
  so `fetchJson<T>` casts the result with `as T` to keep `no-unsafe-*` quiet.
  Clean enough, but worth remembering for any new fetch call.
- No headless browser here, so the visual render is unverified by automation -
  the DOM logic is only exercised by tsc + the e2e serve. Flagged for the user
  to eyeball. A jsdom smoke test would close this gap.

## Lessons

- `web-fetch-json-cast-generic`: with eslint `recommendedTypeChecked`, wrap
  fetches in a `fetchJson<T>` helper that does the single `as T` cast, rather
  than scattering unsafe `any` assignments the linter rejects.
- `frontend-verify-needs-e2e-serve`: a webpack build passing proves compilation,
  not wiring; serve the bundle through the backend and curl `/` + `/api/*` to
  prove the slice actually runs.

## Follow-ups

- Filed: harden `innerHTML` interpolation (escape host-derived strings) + add a
  jsdom render smoke test (LOW, from REVIEW.md) - see the new backlog task.
