# Spike: agent page expansion - sidebar, sessions, context, usage, MCPs

- DATE: 20260719-212152
- STATUS: RECOMMENDED
- TAGS: spike, agent, ui

## Question

The agent (landing) page is a single chat panel. The user wants it to feel like
claude.ai / chatgpt.com: a left sidebar to switch between multiple sessions, a
per-session "context" view (what is in context and how much, like `/context` in
Claude Code), an overall usage indicator (how much of the weekly subscription
limit is left), and "more things" generally - can we add more MCP servers/tools?

The uncertainty this spike reduces: with the `codex exec` subprocess backend
(ChatGPT-subscription auth, no API key), which of these are actually POSSIBLE,
what data does codex already expose, and what new backend/frontend pieces are
needed? A good answer names, per feature, feasible / not-feasible / partial, and
the concrete mechanism.

## Context

The agent backend (`scufris/agent.py`) drives nixpkgs `codex 0.144.4` via
`codex exec [resume <id>]`, parsing the `--json` stream for `thread.started`
(session id), `mcp_tool_call` items, and `turn.completed.usage`. `CodexCliAgent`
holds exactly ONE `thread_id` and resumes it each turn; `reset()` drops it. One
Scufris MCP server (`scufris/mcp_server.py`: `host_stats`, `tatr_ls`, `tatr_show`)
is registered per-invocation via `-c mcp_servers.scufris.*` (nothing written to
`~/.codex`). The frontend agent page (`web/src/agent-view.ts`, `index.html`) is a
single chat panel with a model label, a collapsible tools list, per-turn tool
chips, and a cumulative `ctx N · M out` indicator built from `input_tokens`.

Grounding probe of `$CODEX_HOME` (this host, codex 0.144.4) is what makes this
spike conclusive - codex stores far more than the exec `--json` stream surfaces:

- **Sessions** are JSONL rollout files at
  `$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ts>-<session_id>.jsonl`. Line 1 is a
  `session_meta` payload: `session_id`, `cwd`, `timestamp`, `cli_version`,
  `originator: "codex_exec"`, `source: "exec"`, and `git.branch`. Later lines
  include `user_message` / `agent_message` (usable as a title), `mcp_tool_call_end`,
  and `token_count`.
- **`token_count`** event payload carries everything the "context" and "usage"
  asks need:
  ```
  info.model_context_window: 258400
  info.total_token_usage: {input, cached_input, output, reasoning, total}
  info.last_token_usage:  {...same shape...}
  rate_limits: {
    primary:   { used_percent, window_minutes: 10080, resets_at: <epoch> },
    secondary: null,
    plan_type: "plus", credits: {...}, ...
  }
  ```
  `window_minutes: 10080` = 7 days: this IS the weekly subscription limit, with a
  live `used_percent` and `resets_at`. The context window size (258400) is right
  there too.
- `codex exec resume <SESSION_ID>` resumes any session by UUID (or thread name);
  `--last` picks newest, `--all` disables the default cwd filtering.

So the data for sessions, context %, and weekly-usage already exists locally; the
work is exposing it, not inventing it. This mirrors the lesson
`harvest-the-stream-you-already-run`.

## Options considered

### 1. Session switching (sidebar list + resume)

- **Backend session registry over codex rollouts (RECOMMENDED).** List sessions
  by globbing `$CODEX_HOME/sessions/**/*.jsonl`, reading each line-1
  `session_meta` (id, time, git branch) + first `user_message` as a title;
  filter to this app's sessions by `cwd == server cwd` (matches codex's own
  default resume filter). `CodexCliAgent` gains a settable "current session id"
  (today's single `_thread_id` generalized): new chat -> None (fresh), switch ->
  set to a chosen id, chat -> `resume <id>`. Pros: reuses the exact resume path
  the code already exercises; no new codex behavior; sessions survive restarts
  (they are on disk). Cons: couples to codex's on-disk rollout format (version
  risk); needs cwd/originator filtering so unrelated codex sessions do not leak
  in; concurrent switch during an in-flight turn must respect the existing
  `chat_lock`.
- **Keep session ids only in server memory (rejected).** Track a list of
  thread_ids created this process-lifetime. Simpler, no disk parsing, but loses
  all history on restart and cannot show a title/time - a much weaker "sessions"
  than claude.ai. The disk is right there; use it.
- **Do nothing (rejected).** The single-session panel is what the user is asking
  to move beyond.

### 2. Per-session context view (`/context`-style)

- **Show what codex actually reports (RECOMMENDED, PARTIAL).** From the session's
  last `token_count`: context window (258.4k), used = last `input_tokens`, a
  used% bar, cached vs fresh input tokens, cumulative output/reasoning, turn
  count, and per-tool call counts (already parsed). Pros: all real, all local.
  Cons: this is NOT the full Claude-Code `/context` component breakdown -
  codex does not emit "system prompt = X tok, tools = Y tok, MCP = Z tok,
  messages = W tok". That per-component split is genuinely unavailable from the
  exec backend; be honest and show the axes codex does give (window %, cached
  ratio, tool-call counts) rather than faking a breakdown.
- **Reconstruct a breakdown ourselves (rejected for now).** We could tokenize the
  rollout's message/tool items with a tiktoken-like counter to estimate
  per-component tokens. High effort, model-tokenizer drift, and still an
  estimate; not worth it versus showing codex's real numbers. Revisit only if the
  user specifically wants the component split.

### 3. Overall usage / weekly limit

- **Surface `rate_limits.primary` (RECOMMENDED).** `used_percent` +
  `window_minutes` (10080 = weekly) + `resets_at` -> "weekly limit: 34% used,
  resets in 2d 5h", plus `plan_type` and the secondary window if present. Read it
  from the latest `token_count` in any recent rollout (it is account-wide, not
  session-specific). Pros: exactly the ask, real subscription data. Cons: only
  refreshes when a turn runs (it is emitted mid-turn); a periodic cheap refresh
  would need a turn or a codex status call - acceptable to show "as of last
  turn". Open: whether the exec `--json` STDOUT also carries `token_count`
  (then we would not touch rollout files for usage); the rollout is the reliable
  fallback either way.
- **Do nothing (rejected).** Users on a capped plan want to see headroom.

### 4. More MCP servers / tools ("is it possible? what else is needed?")

- **YES, two complementary axes (RECOMMENDED - do both, cheapest first).**
  - **Expand the Scufris MCP server (cheapest).** Add more read-only
    `@mcp.tool()` handlers to `scufris/mcp_server.py` (e.g. `tatr_show` variants,
    log tailing, `systemctl --user status`, disk/df, git status of a repo) - no
    new process, in the nix closure already. This is the lowest-friction way to
    give the agent "more".
  - **Config-driven multi-server registry.** Generalize `_mcp_overrides` from the
    hard-coded `scufris` block to a list of server specs in `Settings` (id,
    command, args, approval mode), each emitted as `-c mcp_servers.<id>.*`.
    Adding a server (filesystem, git, fetch/websearch) becomes config, not code.
  - What ELSE is needed: each EXTERNAL server needs its binary available on the
    host (nixpkgs / npx / uvx) and a security review of its tools (the Scufris
    guardrail today is "no generic run-any-command tool" + read-only sandbox +
    auto-approve only trusted servers). External servers may want writes/network,
    which fights the read-only sandbox - so gate them behind config and default
    off. codex ALSO has its own skills/plugins system under `$CODEX_HOME`
    (skill-installer, imagegen, ...), but that is codex-managed and not the clean
    per-invocation injection path our MCP registration uses; leave it out of
    scope.
- **Rejected: a generic "run any shell command" tool.** Explicitly against the
  server's security model. Curated handlers only.

### Sidebar layout (cross-cutting)

- **RECOMMENDED: a two-pane shell on the agent page.** Left sidebar (fixed width,
  collapsible on narrow screens) = new-chat button + session list + a compact
  weekly-usage meter; main pane = the existing chat + a context strip/panel. This
  mirrors claude.ai/chatgpt and is a pure `index.html` + CSS + `agent-view.ts`
  restructure; the stats page is untouched.

## Recommendation

Build all four, in dependency order, as separate `/flow`s:

1. **Backend first (tatr 20260719-212203)** - the data layer everything else
   consumes: a multi-session registry (list/switch/new over codex rollouts), a
   per-session context object (window, token usage, tool-call counts), and an
   account usage/quota object (`rate_limits.primary`). New endpoints:
   `GET /api/agent/sessions`, `POST /api/agent/session` (switch/new),
   `GET /api/agent/context`, `GET /api/agent/usage`. Generalize `CodexCliAgent`'s
   single `_thread_id` into a settable current-session id.
2. **Sidebar + session switching (tatr 20260719-212205)** - the two-pane shell,
   session list, new-chat, click-to-switch (reloads that session's transcript if
   we choose to render history; at minimum switches the resume target).
3. **Context + weekly-usage panel (tatr 20260719-212207)** - the `/context`-style
   panel (window %, cached ratio, tool counts) + the weekly-limit meter
   (used_percent, resets_at, plan_type), honest about what codex does/doesn't
   expose.
4. **MCP reach (tatr 20260719-212208)** - expand the Scufris server's tools AND
   make servers config-driven so more can be added; external servers gated + off
   by default.

Feasibility verdict: sessions, context %, and weekly usage are FULLY feasible
(data already on disk); the only genuine limitation is the fine-grained
per-component `/context` breakdown, which the exec backend does not expose -
show codex's real axes instead. More MCPs are feasible; the "something else"
needed is each external server's binary + a security gate, so the cheap win is
adding tools to the Scufris server we already run.

## Open questions

- Does `codex exec --json` STDOUT emit `token_count` (with `rate_limits`)? If yes,
  usage/context can be harvested from the stream we already read, no rollout-file
  parsing for the live turn. Resolve with one captured `--json` run; the rollout
  file is the fallback regardless (and is needed for session listing anyway).
- Session-list scoping: filter by `cwd == server cwd` vs `originator/source`.
  cwd matches codex's own resume filter; confirm the server's runtime cwd is
  stable (it is, uvicorn process cwd) and pick the filter during backend /plan.
- Switching a session: do we re-render that session's PAST transcript in the chat
  log (parse `user_message`/`agent_message` from the rollout) or just retarget
  resume and start appending? claude.ai shows history; rendering it is a nice-to-
  have that the backend session object can support - a /plan call.
- Usage freshness: `rate_limits` updates only when a turn runs. Acceptable to
  label "as of last turn"; a forced refresh would cost a turn.

## Next steps

Direction-level tasks seeded (for `/plan` to break into steps):

- tatr 20260719-212203: Agent backend - multi-session registry + context & usage/quota endpoints
- tatr 20260719-212205: Agent page - left sidebar with session list + switching
- tatr 20260719-212207: Agent page - context breakdown + weekly-usage panel
- tatr 20260719-212208: Agent reach - config-driven MCP server registry + more Scufris tools

## Fix record

(Appended by each implementing task as it lands.)
