# F1: SPA dynamic routing + fallback + the /agents/<id> agent-detail page shell

- STATUS: OPEN
- PRIORITY: 44
- TAGS: agents,frontend


## Goal

Introduce real per-agent routing: a FastAPI catch-all that serves the SPA shell
for `/agents/<id>` (and `/agents/<id>/settings`) when the path is not a static
asset, so client-side routing works; add the webpack `agent-detail` entry +
`historyApiFallback` for `/^\/agents\//`. This is the structural gate for the
per-agent page (F3) and chat (F4).

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (decision 1; recommendation F1). The MPA/no-fallback gap is the
  single biggest structural blocker.
- No hard dep, but pairs with F2/F3.
