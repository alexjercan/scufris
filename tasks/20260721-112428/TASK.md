# F0: quick UI polish bugs (SSE reattach on select, status poll interval, empty states)

- STATUS: OPEN
- PRIORITY: 52
- TAGS: agents,ux,frontend


## Goal

Quick, independent frontend polish (shippable first, no backend change):
- Reattach the SSE EventSource when SELECTING an already-running agent (today it
  only opens on Run); the events endpoint replays via Last-Event-ID.
- Add a modest status `setInterval` while a running agent is open, so
  turns/tokens refresh even between SSE events (mirror stats-view polling).
- Empty/guidance states: "create a project first" when the agent create form has
  no projects (disable submit); show "not started" instead of 0s for a never-run
  agent.

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (current-state review, bugs m2/m3/m5).
- No deps; can land before the refactors.
