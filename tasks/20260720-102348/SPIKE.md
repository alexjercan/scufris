# Spike: agent-page UX review round 2 (head, tools, affordances, settings)

- DATE: 20260720-102348
- STATUS: RECOMMENDED
- TAGS: spike, agent, ui

## Question

The agent page has changed a lot since the first UX spike (20260719-223054):
markdown rendering, token streaming, a "thinking" section, a multi-line composer,
grouped sidebar sections, and message affordances (copy/timestamps/scroll/
onboarding/a11y) all landed. With fresh, adversarial eyes: what still reads as
unpolished or wrong, given a technical homelab operator as the user? Specifically
- is the chat head a good use of space, are tools surfaced well, are the message
affordances discoverable, and would a read-only settings/config view help? The
answer is a prioritized set of direction-level tasks.

## Context

The agent page (`web/src/index.html`, `agent-view.ts`, `style.css`, `markdown.ts`;
nav in `_header.html`) is a two-pane shell: a left sidebar (New chat, a scrolling
session list, and three labeled stat boxes - Sessions / this session / account)
and a right chat pane. The chat pane's head is `<h2>Agent</h2>` + `model gpt-5.5`
+ a `tools` toggle button (`index.html` chat**head / `agent-bar`); the toggle
reveals a raw name+description list (`renderAgentPanel`, `agent-view.ts`). Message
footers carry a timestamp plus a copy (assistant) or edit (user) button that are
`opacity: 0` until `.chat__foot:hover` (`style.css`). All config is
environment-variable only and read-only in the UI; `config.py` already holds the
knobs (`agent_backend` app_server/exec/mock, `agent_model`, `agent_auth_mode`,
`agent_tools_enabled`, `mcp_servers`, sandbox is always read-only) and
`/api/config` + `/api/agent/info` + `/api/agent/tools` already expose some of it.
The agent runs codex with a scufris MCP server (`mcp_server.py`: `host_stats`,
`disk_usage`, `list_processes`, `tatr_ls/show/new`) AND a read-only shell;
`_exec_args` (`agent.py`) passes only the user prompt - no instructions steer the
model toward the MCP tools, so "tell me about this host" runs `uname`/`df` in the
shell instead of `host_stats`.

Inputs to this spike: the user's own playtest feedback plus a fresh adversarial
review by an unbiased subagent that had not seen the implementation.

## Options considered

The "options" here are the candidate problems and whether each earns a task. What
is genuinely good is kept; the rest converges into prioritized tasks.

### What is good (keep it)

- Markdown rendering (safe DOM build, fenced code + per-block copy), token-by-token
  streaming with a spinner/elapsed/ran-tools status line, the collapsible thinking
  section, timestamp formatting, and the fork/edit-to-branch flow are all solid.
  The problems below are discoverability and control, not the core chat loop.

### What is bad (the real pain)

- **Message affordances are hover-only (user's headline complaint).** Copy on a
  reply and edit on a user turn are `opacity: 0` until the footer is hovered - so
  they are invisible on touch, and easy to miss with a mouse (the footer is a thin
  strip). Copying a reply is table stakes; it must be visible. (user + review BAD)
- **The agent bypasses the curated scufris tools for raw shell (user's #2).** With
  no steering instructions, codex answers host questions with `uname`/`df` shell
  calls instead of `host_stats`/`disk_usage`/`list_processes`. That loses the tool
  chips, is slower/less predictable, and makes the whole MCP surface look unused.
  Root cause confirmed in `agent.py:_exec_args` (prompt only, no instructions).
- **The chat head wastes prime space on a redundant "Agent" title (user's #3).**
  `_header.html`'s nav already has an active "Agent" link, yet the chat head
  repeats `<h2>Agent</h2>`, then crams `model gpt-5.5` + a `tools` toggle into one
  bar. The head should answer "what model, how full is my context, is the agent
  ready" at a glance - not restate the page name.
- **Tools are a bare, afterthought list (user's #4).** The toggle dumps
  name+description in a rigid 2-column grid with no separation, no status (enabled/
  disabled), no grouping, and no scroll container (it can shove the log off-screen).
  For a homelab operator, "what can it do / what did I enable" is core context and
  currently looks like debug output.

### What is mediocre (rough edges)

- Tool-call chips after a reply are small/low-contrast with a negative-margin hack
  - tool execution is the point of the agent and is visually de-emphasized.
- No read-only settings/config surface at all: backend, model, auth mode, sandbox,
  enabled tools, and MCP servers are invisible in the UI, so a user cannot see or
  understand their setup, or why the agent is disabled, without reading env/logs.
- Onboarding examples and the composer placeholder are hardcoded and never mention
  fork/edit or actual capabilities; session titles truncate with no full-title
  tooltip; the "new messages" pill does not say how many.

### Do nothing

Viable for the core chat loop (it is good). Not viable for the affordance and
tool-steering issues: those are daily friction the user explicitly hit. The head/
tools/settings items are polish-with-real-value and can be sequenced after.

## Recommendation

Fix the two friction bugs first, then the head/tools presentation, then the
settings surface, then polish. Five direction-level tasks:

1. **Message affordances always-visible (P40).** Copy (reply) and edit (user) are
   persistently visible (dimmed, brightening on hover/focus) and work on touch +
   keyboard; a reply-level copy that is obvious. Small, high-value, the user's #1.
2. **Steer the agent to prefer the scufris MCP tools over shell (P40).** Give codex
   instructions (a base-instructions/preamble via `-c`, an instructions file, or a
   prompt prefix) so host/tatr questions use `host_stats`/`disk_usage`/
   `list_processes`/`tatr_*` rather than raw `uname`/`df`. Verify on a live
   "tell me about this host" that a tool chip appears. Behavioral; gate honestly.
3. **Chat head redesign (P30).** Drop the redundant `<h2>Agent</h2>`; slim the head
   to actionable at-a-glance info (model, context fill/status), and remove the raw
   `tools` toggle from the head - its entry point moves to the settings view (task
   4). Pure frontend.
4. **Read-only settings/config view + nicer tool presentation (P30).** A read-only
   page/panel showing backend, model, auth mode, sandbox (read-only), enabled-tools
   state, and MCP servers, with the tool list rendered as proper cards (name,
   description, source/server, maybe category) - this is the new, nicer home for
   the tools moved off the head. Likely needs a small `/api/agent/config` (or an
   extended `/api/agent/info`) aggregating the knobs. Read-only now; the "editable
   settings / change the LLM" idea is explicitly deferred (a later spike/task).
5. **Agent-page discoverability polish (P20).** Make tool-call chips prominent
   ("ran: host_stats"); add a full-title tooltip on session rows; put a count on
   the "new messages" pill; hint at fork/edit in the onboarding or placeholder. A
   grab-bag of the mediocre findings; do any order after 1-4.

Do 1 and 2 first (they are the friction the user actually hit). 3 and 4 are
coupled (the head loses the tools; the settings view gains them), so plan 4's
tool-card presentation before finalizing 3's head. 5 is optional cleanup.

## Open questions

- **Settings view shape (task 4): new nav page vs a panel.** A third nav entry
  (`/settings/`) mirrors the existing multi-page webpack setup (one entry + one
  HtmlWebpackPlugin per page, per `webpack-multipage-htmlplugin-per-page`), but a
  slide-over panel on the agent page keeps context. Decide at /plan; a page is more
  discoverable and reuses the existing pattern.
- **Tool-steering mechanism (task 2): which codex lever actually works.** Options:
  `-c` base/experimental instructions, an instructions file, or a prompt preamble
  we prepend. Must PROBE on the target host which one codex honors without dropping
  the read-only sandbox (per `probe-runtime-on-target-host-early`); a preamble is
  the low-risk fallback if the config levers are ignored.
- **Editable settings later.** The user wants eventual enable/disable + model
  switching. That is a separate spike (writing config back / restarting the agent
  cleanly) once the read-only view exists - not in scope here.

## Next steps

Direction-level tasks seeded (for `/plan` to break into steps):

- tatr 20260720-102558 (P40): message affordances always-visible (copy/edit, touch+kbd)
- tatr 20260720-102559 (P40): steer the model to prefer the scufris MCP tools over shell
- tatr 20260720-102600 (P30): chat head redesign (drop the duplicate title, slim it)
- tatr 20260720-102601 (P30): read-only settings/config view + nicer tool presentation
- tatr 20260720-102602 (P20): agent-page discoverability polish (chips/tooltip/pill/hints)

Suggested order: 102558 and 102559 first (the friction the user actually hit),
then 102600 + 102601 together (the head loses the tools; the settings view gains
them), then 102602 (optional cleanup).

## Fix record

- 20260720-102558 (message affordances always-visible, P40) - LANDED. Copy/edit
  resting `opacity: 0` -> `0.6`, brightening on hover/focus, so they are visible on
  touch and never hover-hidden. CSS-only; 73 frontend tests green. See
  tasks/20260720-102558/TASK.md.
- 20260720-102559 (steer tools over shell, P40) - LANDED. codex ignores tool
  descriptions / instructions files / AGENTS.md for tool choice (probed live); only
  a preamble on the turn prompt works. Prepend a sentinel-wrapped `STEERING_PREAMBLE`
  (both backends), strip it from titles/transcripts. Verified e2e: host questions
  now call host_stats/disk_usage/list_processes, 0 shell. 129 pytest + 73 frontend.
  See tasks/20260720-102559/TASK.md.
- 20260720-102601 (read-only settings/config view, P30) - LANDED. New `/settings/`
  nav page (multipage pattern) showing status/backend/model/auth/sandbox/tools +
  MCP servers + tool cards, from a new `GET /api/agent/config`; says "tools
  disabled" instead of listing a dead catalog. 131 pytest + 79 frontend green,
  verified e2e. The new home for the tools; unblocks 102600. See
  tasks/20260720-102601/TASK.md.

- 20260720-102600 (chat head redesign, P30) - LANDED. Dropped the redundant
  `<h2>Agent</h2>` and the tools toggle/inline panel; the head is now a slim row
  (model + a compact "N tools" link to `/settings/`). Net -94 lines. 79 frontend
  tests green. Completes the head+settings pair. See tasks/20260720-102600/TASK.md.

Round-2 remaining: 102602 (discoverability polish) - the last one.
