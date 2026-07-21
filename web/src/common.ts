// Shared types and DOM/fetch helpers used by both pages (agent + stats).

export interface MemStats {
    total: number;
    used: number;
    available: number;
    percent: number;
}

export interface SwapStats {
    total: number;
    used: number;
    percent: number;
}

export interface DiskUsage {
    mountpoint: string;
    total: number;
    used: number;
    percent: number;
}

export interface NetIO {
    bytes_sent: number;
    bytes_recv: number;
}

export interface GpuStats {
    name: string;
    util_percent: number;
    mem_used_mb: number;
    mem_total_mb: number;
    mem_percent: number;
    temp_c: number;
    power_w: number;
    power_limit_w: number;
    clock_sm_mhz: number;
    clock_mem_mhz: number;
}

export interface SensorReading {
    label: string;
    current: number;
    high: number | null;
    critical: number | null;
}

export interface SensorGroup {
    chip: string;
    readings: SensorReading[];
}

export interface FanReading {
    chip: string;
    label: string;
    rpm: number;
}

export interface NetIfRate {
    name: string;
    sent_per_sec: number;
    recv_per_sec: number;
}

export interface DiskIoRate {
    name: string;
    read_per_sec: number;
    write_per_sec: number;
}

export interface CpuActivity {
    ctx_switches_per_sec: number;
    interrupts_per_sec: number;
}

// Mirrors scufris.metrics.HostStats (the /api/stats payload).
export interface HostStats {
    hostname: string;
    os_name: string;
    kernel: string;
    cpu_percent: number;
    per_cpu_percent: number[];
    mem: MemStats;
    swap: SwapStats;
    disks: DiskUsage[];
    load_avg: [number, number, number];
    uptime_seconds: number;
    net: NetIO;
    sampled_at: string;
    gpus: GpuStats[];
    temps: SensorGroup[];
    fans: FanReading[];
    per_cpu_freq_mhz: number[];
    net_interfaces: NetIfRate[];
    disk_io: DiskIoRate[];
    process_count: number;
    cpu_activity: CpuActivity;
}

export interface AppConfig {
    poll_seconds: number;
    agent_enabled: boolean;
}

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

export type StreamEvent =
    | StreamToolEvent
    | StreamDoneEvent
    | StreamErrorEvent
    | StreamTextDeltaEvent
    | StreamReasoningDeltaEvent;

export interface AgentInfo {
    model: string;
    auth_mode: string;
    enabled: boolean;
}

export interface AgentTool {
    name: string;
    description: string;
    server: string;
    args: string[];
    enabled: boolean;
}

export interface McpServerInfo {
    id: string;
    source: string; // "built-in" | "configured"
}

export interface AgentConfig {
    enabled: boolean;
    backend: string;
    model: string;
    auth_mode: string;
    tools_enabled: boolean;
    sandbox: string;
    mcp_servers: McpServerInfo[];
    writable: boolean;
}

// One extra MCP server the operator can add from the settings page.
export interface McpServerSpec {
    id: string;
    command: string;
    args?: string[];
    approve?: boolean;
}

// A whitelisted, partial config change sent to PATCH /api/agent/config.
export interface AgentConfigUpdate {
    agent_enabled?: boolean;
    agent_backend?: string;
    agent_model?: string;
    agent_tools_enabled?: boolean;
    disabled_tools?: string[];
    mcp_servers?: McpServerSpec[];
}

export interface HealthCheck {
    name: string;
    status: string; // "ok" | "warn" | "error"
    detail: string;
    hint: string;
}

export interface AgentHealth {
    scufris_version: string;
    codex_version: string | null;
    session_count: number;
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
    auth_mode: string;
    model: string;
    enabled: boolean;
    quota: UsageQuota | null;
}

// Named config profiles and which one is active.
export interface ProfilesResponse {
    profiles: string[];
    active: string;
}

// A first-class project (mirrors scufris.projects.Project).
export interface Project {
    id: string;
    cwd: string;
    name: string;
    language: string;
    description: string;
}

// One tatr task belonging to a project (its specs).
export interface ProjectTask {
    id: string;
    title: string;
    priority: number;
    tags: string[];
}

// A configured agent (mirrors the backend AgentRecord). Bound to a project via
// project_id; `state` is the run lifecycle; `permission_mode` the write posture.
export interface Agent {
    id: string;
    name: string;
    project_id: string;
    backend: string;
    model: string;
    goal: string;
    task_id: string;
    session_id: string | null;
    state: string;
    permission_mode: string;
}

// The write postures an agent can run in (Claude-style), default "manual".
export const PERMISSION_MODES = ["manual", "edit", "auto"];

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
}

// The user-facing backends an agent can select. "mock" is dev-only (behind a
// server flag) and not offered here; legacy "app_server"/"exec" collapse to
// "codex" on the backend. Friendly display labels land with the cards (F2).
export const AGENT_BACKENDS = ["codex", "claude"];

export const DEFAULT_POLL_SECONDS = 2;

// Escape a host-derived string before it goes into innerHTML. Numbers (percent,
// bytes) are formatted via toFixed and cannot inject markup, but strings such as
// a disk mountpoint or hostname could, so every interpolated string is escaped.
export function escapeHtml(value: string): string {
    return value
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

export function el(
    tag: string,
    className?: string,
    html?: string,
): HTMLElement {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (html !== undefined) node.innerHTML = html;
    return node;
}

export async function fetchJson<T>(url: string): Promise<T> {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`${url} -> ${String(resp.status)}`);
    return (await resp.json()) as T;
}

// Send a JSON body with a method (PATCH/POST/DELETE) and parse the JSON reply.
// On a non-2xx it throws an Error carrying the server's `detail` when present, so
// a caller can surface a clear message (e.g. a 422 for a bad MCP id).
export async function sendJson<T>(
    url: string,
    method: string,
    body?: unknown,
): Promise<T> {
    const resp = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!resp.ok) {
        let detail = `${url} -> ${String(resp.status)}`;
        try {
            const data = (await resp.json()) as { detail?: string };
            if (data.detail) detail = data.detail;
        } catch {
            // non-JSON error body; keep the status-based message
        }
        throw new Error(detail);
    }
    return (await resp.json()) as T;
}

export function formatBytes(bytes: number): string {
    if (bytes <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB", "PB"];
    const exp = Math.min(
        units.length - 1,
        Math.floor(Math.log(bytes) / Math.log(1024)),
    );
    const value = bytes / Math.pow(1024, exp);
    return `${value.toFixed(exp === 0 ? 0 : 1)} ${units[exp]}`;
}

export async function loadConfig(): Promise<AppConfig> {
    try {
        return await fetchJson<AppConfig>("/api/config");
    } catch {
        return { poll_seconds: DEFAULT_POLL_SECONDS, agent_enabled: false };
    }
}
