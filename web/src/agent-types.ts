// The chat, agent, session, usage and project wire shapes shared by the agent
// pages.

export interface ToolCall {
    server: string;
    tool: string;
    status: string;
}

export interface TokenUsage {
    input_tokens: number;
    cached_input_tokens: number;
    output_tokens: number;
    reasoning_output_tokens: number;
}

export interface ChatReply {
    text: string;
    status?: string;
    tool_calls: ToolCall[];
    usage: TokenUsage | null;
}

// An image attached to a chat turn (base64 payload + MIME), sent to the backend
// which writes it to a temp file for codex to see.
export interface ImageAttachment {
    data_base64: string;
    mime: string;
}

// Live turn-progress events streamed over SSE from /api/chat/stream.
export interface StreamToolEvent {
    kind: "tool";
    tool: ToolCall;
}

export interface StreamDoneEvent {
    kind: "done";
    reply: ChatReply;
    session_id: string | null;
}

export interface StreamErrorEvent {
    kind: "error";
    detail: string;
}

// app-server backend only: token-by-token text + reasoning ("thinking").
export interface StreamTextDeltaEvent {
    kind: "text_delta";
    delta: string;
}

export interface StreamReasoningDeltaEvent {
    kind: "reasoning_delta";
    delta: string;
}

// Emitted at turn-start (codex) the moment the session id is known, before the
// reply streams, so the client learns the in-flight session without waiting for
// `done`. Lets a tab that started a turn pin the new session id live.
export interface StreamSessionStartedEvent {
    kind: "session_started";
    session_id: string;
}

export type StreamEvent =
    | StreamToolEvent
    | StreamDoneEvent
    | StreamErrorEvent
    | StreamTextDeltaEvent
    | StreamReasoningDeltaEvent
    | StreamSessionStartedEvent;

export interface AgentInfo {
    model: string;
    auth_mode: string | null; // null for a backend with no login (mock)
    enabled: boolean;
}

export interface ToolParam {
    name: string;
    type: string; // JSON-schema type: string | integer | number | boolean | ...
    required: boolean;
    description: string;
    default: unknown;
}

export interface AgentTool {
    name: string;
    description: string;
    server: string;
    args: string[];
    parameters: ToolParam[]; // full param schema, for the "try it" runner
    enabled: boolean; // false when the operator disabled it (disabled_tools)
    available?: boolean; // false when its server is unhealthy (live-probe verdict)
}

// One scufris MCP server's live-probe result for the settings "MCP tools"
// section (from GET /api/agent/mcp or /api/agents/{id}/mcp). `status` drives the
// per-server dot (green/amber/red); each tool carries `enabled` + `available` for
// its bulb.
export interface McpServerHealth {
    id: string; // scufris | den | agent
    status: "ok" | "warn" | "error";
    detail: string;
    tools: AgentTool[];
}

// The result of running one MCP tool via POST /api/agent/tools/{name}/run.
export interface ToolRunResult {
    ok: boolean;
    text: string;
    structured: Record<string, unknown>;
}

export interface AgentConfig {
    enabled: boolean;
    backend: string;
    model: string;
    auth_mode: string | null;
    tools_enabled: boolean;
    sandbox: string;
    writable: boolean;
}

// A whitelisted, partial config change sent to PATCH /api/agent/config.
export interface AgentConfigUpdate {
    agent_enabled?: boolean;
    agent_backend?: string;
    agent_model?: string;
    agent_tools_enabled?: boolean;
    disabled_tools?: string[];
}

export interface HealthCheck {
    name: string;
    status: string; // "ok" | "warn" | "error"
    detail: string;
    hint: string;
}

export interface AgentHealth {
    scufris_version: string;
    // The effective backend probed + its CLI version - neutral so a claude agent
    // reports claude, not codex.
    backend: string;
    backend_version: string | null;
    // null when no reading was taken - the backend has no session reader, or the
    // agent is disabled. A number is a real reading, 0 included.
    session_count: number | null;
    last_session: string | null;
    checks: HealthCheck[];
}

export interface SessionInfo {
    id: string;
    title: string;
    started_at: string | null;
    updated_at: string | null;
    git_branch: string | null;
    cwd: string | null;
}

export interface SessionsResponse {
    sessions: SessionInfo[];
    current: string | null;
}

export interface TranscriptMessage {
    role: string;
    text: string;
    ts: string | null;
    // Assistant turns carry the tools they ran + the turn's usage, so the chips
    // and token count re-render on reload (empty/null for user turns).
    tool_calls: ToolCall[];
    usage: TokenUsage | null;
    // Codex "thinking" recovered from the sidecar, so the collapsed spoiler
    // survives a hard reload (null when absent: user turns, non-codex, or turns
    // the sidecar does not cover).
    reasoning: string | null;
}

export interface SessionContext {
    session_id: string;
    context_window: number;
    input_tokens: number;
    cached_input_tokens: number;
    output_tokens: number;
    reasoning_output_tokens: number;
    total_tokens: number;
    turn_count: number;
    tool_call_count: number;
}

export interface RateWindow {
    used_percent: number;
    window_minutes: number;
    resets_at: number | null;
}

// One diagnostic answer plus whether the agent's backend can answer it at all
// (mirrors scufris.backends.base.Capability). `supported: false` means the
// backend has no such reader - not that the reader found nothing.
export interface Capability<T> {
    supported: boolean;
    value: T | null;
}

export interface UsageQuota {
    plan_type: string | null;
    primary: RateWindow | null;
    secondary: RateWindow | null;
}

// The agent's persistent footprint on disk (codex rollouts).
export interface MemoryFootprint {
    session_count: number;
    total_bytes: number;
    oldest: string | null;
    newest: string | null;
}

// The account backing the agent (for the Account panel).
export interface AccountInfo {
    auth_mode: string | null;
    model: string;
    enabled: boolean;
    quota: Capability<UsageQuota>;
}

// A first-class project (mirrors scufris.projects.Project).
export interface Project {
    id: string;
    cwd: string;
    name: string;
    language: string;
    description: string;
}

// A candidate directory for the Projects page: a discovered dir, a registered
// project, or both. `registered`/`project_id` mark ones already tracked.
export interface DiscoveredProject {
    path: string;
    name: string;
    language: string;
    registered: boolean;
    project_id: string | null;
}

// The Projects page payload: discovered-union-registered dirs + the base dirs
// offered in the create form's picker (GET /api/projects/discovered).
export interface DiscoveredProjects {
    projects: DiscoveredProject[];
    base_dirs: string[];
}

// One tatr task belonging to a project (its specs).
export interface ProjectTask {
    id: string;
    title: string;
    priority: number;
    tags: string[];
}

// One project-defined SKILL.md recipe (mirrors backend ProjectSkill). `source`
// is the file path relative to the project cwd. Read-only. (The backend defaults
// `description` to "" and always emits it, so it is required here.)
export interface ProjectSkill {
    name: string;
    description: string;
    source: string;
}

// One project-defined MCP server / custom tool (mirrors backend ProjectTool).
// `kind` is the transport ("stdio"/"http"/..., or "" when unknown); `source` is
// the config file path relative to the project cwd. Read-only.
export interface ProjectTool {
    name: string;
    description: string;
    source: string;
    kind: string;
}

// The read-only capability surface of an agent's project (GET
// /api/agents/{id}/capabilities). Empty for the orchestrator / a project-less
// agent. Mirrors backend ProjectCapabilities.
export interface ProjectCapabilities {
    skills: ProjectSkill[];
    tools: ProjectTool[];
}

// A configured agent (mirrors the backend AgentRecord). Bound to a project via
// project_id; `state` is the run lifecycle; `permission_mode` the write posture.
export interface Agent {
    id: string;
    name: string;
    project_id: string;
    backend: string;
    model: string;
    description: string;
    goal: string;
    task_id: string;
    session_id: string | null;
    state: string;
    permission_mode: string;
}

// The merged live run-state + backend progress for one agent (GET .../status).
export interface AgentRunStatus {
    agent_id: string;
    state: string;
    session_id: string | null;
    turns: number;
    tool_calls: number;
    input_tokens: number;
    output_tokens: number;
    context_window: number;
    last_message: string | null;
    updated_at: number | null;
    // The in-flight turn's prompt (steering stripped), present only while the run
    // is live, so a mid-turn reattach can render the user bubble the transcript
    // has not caught up on yet. null/absent when idle.
    prompt?: string | null;
}

// One selectable backend from GET /api/agents/backends: the server is the
// source of truth for which backends are available (mock only when its dev flag
// is on) and each backend's default model, so the pickers cannot drift.
export interface BackendOption {
    id: string;
    label: string;
    default_model: string;
    // The suggested model catalog for this backend (autocomplete); the model
    // field still accepts a free-text value not in the list.
    models: string[];
}
