# Spike: Dashboard app style and layout

- PRIORITY: 0
- TAGS: spike, backlog, dashboard, ui
- KIND: SPIKE
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Question

What should the Scufris dashboard be built as - a web SPA served by the
existing FastAPI backend, a terminal UI (TUI), or a native/desktop shell - and
what layout best presents live host stats alongside a CLI-control panel and a
chat panel? Land on one recommended direction with a concrete first-screen
layout.

## Context

`pyproject.toml` already pulls in FastAPI + Uvicorn and rich, which biases
toward either a web frontend fronted by FastAPI or a rich-based TUI. The app
is single-host and personal (one NixOS machine), so multi-user concerns are
out of scope. Three surfaces must coexist: a metrics dashboard, a CLI-control
panel, and a chat panel (see the agent-harness spike).

## What a good answer looks like

A recommended UI approach with the runner-up honestly weighed, plus a rough
layout sketch for the first screen (where metrics / CLI controls / chat live),
and a note on the live-update transport (WebSocket / SSE / poll). Concrete
enough that `/plan` can turn it into steps without re-litigating the choice.

## Candidate directions to explore (diverge before converging)

- **Web SPA behind FastAPI** - HTMX/Alpine, or a JS framework, or server-driven
  components; SSE/WebSocket for live metrics. Pros: rich charts, reuses the
  FastAPI surface. Cons: a frontend build/stack to maintain.
- **Rich/Textual TUI** - runs in the terminal, no browser. Pros: light, native
  to a headless box, rich already present. Cons: charts and layout are more
  constrained; chat UX is rougher.
- **Native/desktop shell** (pywebview, Tauri-sidecar, etc.) - a window around
  the web UI. Pros: feels like an app. Cons: extra packaging via nix.
- **Do nothing yet** - ship a JSON/metrics API first and defer the UI.

## Notes

- Output per the /spike skill: write `tasks/<id>/SPIKE.md`, seed direction-level
  tasks, close this spike task.
- Keep the live-update mechanism a single swappable seam so a transport choice
  can be reversed cheaply.
- Depends on nothing; unblocks the metrics-collection and CLI-control work.
