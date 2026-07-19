# Add agent chat panel to the dashboard (streaming)

- STATUS: OPEN
- PRIORITY: 15
- TAGS: feature,backlog,agent,ui

## Goal

Add the agent chat panel to the dashboard: chat with the Scufris agent from the
UI, with streaming replies.

## Notes

- Spike: tasks/20260719-153040/SPIKE.md.
- Frontend: a chat panel in the existing web/ single-page app (web/src), styled
  with the existing scufris theme; sends messages to a backend chat endpoint and
  renders streaming assistant replies (SSE or chunked). Keep the transport a
  single swappable seam like the stats polling.
- Backend: a FastAPI chat endpoint that drives the Agent interface from
  tatr 20260719-162356 (turns/threads); stream tokens back to the client.
- Show tool activity when the agent runs a tool (e.g. a tatr command) so the
  user can see what it did (ties to the MCP tool server, tatr 20260719-162419).
- Depends on the agent backend (tatr 20260719-162356). Pairs with the tool
  server for visible tool calls.
