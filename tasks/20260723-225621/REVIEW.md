# Review: Render read-only project skills+tools cards on the agent settings page

- TASK: 20260723-225621
- BRANCH: feature/project-capabilities-ui

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

- [x] R1.1 (NIT) web/src/common.ts:372 - `ProjectSkill.description` /
  `ProjectTool.description` are typed as required `string`, but the backend
  Pydantic models declare `description: str = ""` (defaulted). Not wrong at
  runtime (the server always emits the field), but consider a comment noting the
  backend default.
  - Response: fixed - added a comment on the `ProjectSkill` interface noting the
    backend defaults `description` to "" and always emits it, so required here is
    correct. Kept the type required (better for the render code).

Verification (in-session supplement): re-derived the load-bearing behavior
myself - the null-capabilities fetch skip (`isOrchestrator || !agent?.project_id`
-> null) lines up with the `if (data.capabilities)` render guard, so a
project-less agent renders NEITHER card while a project agent with empty lists
renders explicit "none" cards; watched the three new vitest cases fail before the
render functions existed and pass after (would fail if the render were removed).
`npm run ci` (format:check + lint + vitest 20/20 + webpack build) green. Field
names/types mirror scufris/project_capabilities.py exactly; all interpolated
values are `escapeHtml`-escaped (load-bearing since `el()` assigns via
innerHTML). Non-ASCII DoD grep hits (agent-settings-view.ts:111,265) confirmed
pre-existing and untouched by the diff. No open `manual:` DoD items on this task;
the goal-level manual acceptance (real project agent shows its skills/tools) is
batched to the umbrella GOAL.md for the flow Finish gate.
