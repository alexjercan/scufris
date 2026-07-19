# Spike: Host metrics collection approach

- STATUS: CLOSED
- PRIORITY: 0
- TAGS: spike, backlog, dashboard, monitoring

## Question

How should Scufris collect host metrics on a Linux/NixOS machine - which stats
matter, and via which source (a Python library like psutil, reading
/proc + /sys directly, or shelling out to system tools) - and what internal
shape should a metric take so both the dashboard and the agent can consume it?

## Context

The monitoring dashboard is the first Scufris pillar and the dashboard-style
spike depends on having something to render. The target host is NixOS, so any
external binary must be available in the nix dev shell / runtime closure. The
same metrics feed the chat agent as a tool it can query.

## What a good answer looks like

A recommended collection approach with the runner-up weighed on: coverage
(CPU, mem, swap, disk usage + IO, network, load, processes, temps/sensors,
uptime), NixOS/closure friendliness (pure-Python vs extra system deps), polling
cost, and testability (can a collector be faked in tests). Plus a small
proposed pydantic model / interface for a metric sample so the dashboard and
agent share one shape.

## Candidate directions to explore (diverge before converging)

- **psutil** (or similar pip lib) - cross-cutting, pure-ish Python, easy to
  test; check it packages cleanly through uv2nix on NixOS.
- **Direct /proc and /sys reads** - zero deps, full control, more parsing code
  and Linux-specific.
- **Shell out to system tools** - reuse `free`, `df`, `sensors`, `ip`, etc.;
  needs them in the nix closure and parsing their output is brittle.
- **A mix** - a library for the common case, direct reads/tools for gaps
  (e.g. sensors).

## Notes

- Output per the /spike skill: write `tasks/<id>/SPIKE.md`, seed direction-level
  tasks, close this spike task.
- Keep collection behind one interface so a sampler can be faked in tests
  (harness-first) and a source swapped without touching the dashboard.
- Depends on nothing hard; pairs with the dashboard-style spike.
