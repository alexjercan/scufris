# Spike: token-by-token streaming + reasoning + events via codex

- PRIORITY: 0
- TAGS: spike, agent
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Research whether token-by-token streaming + reasoning ("thinking") + live tool/
other events are obtainable from codex, and how. Deliverable is the research doc
+ seeded tasks.

## Outcome

RECOMMENDED. Proven by probing real turns: `codex exec --json` is turn-level with
ZERO token deltas (and rollout files are completed-item granularity too). The
`codex app-server` (experimental) JSON-RPC protocol IS the streaming path - its
schema defines `outputDelta`, `ReasoningTextDelta`, thread-item/tool/plan/process
events. So token-by-token + reasoning + all events require migrating to
`app-server`. See tasks/20260720-002611/SPIKE.md.

Seeded: 20260720-002619 (app-server streaming backend behind the Agent seam,
probe-first, config-gated so exec stays a fallback), 20260720-002621 (chat UI:
token-by-token text, thinking section, event feed). Backend first.

Caveat surfaced: app-server is EXPERIMENTAL (protocol churn); a large
rearchitecture. Confirm the user accepts the experimental dependency before the
full build.
