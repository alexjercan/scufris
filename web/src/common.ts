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
    // The host overview shells out (systemctl, nixos-rebuild), so it polls on
    // its own much slower clock rather than riding the 2s stats poll.
    host_overview_seconds: number;
}

// --- host inspection (/api/host/overview) -----------------------------------
//
// Mirrors scufris/host: every report carries its own availability, so the UI
// renders a REASON when something could not be read instead of an empty card
// that reads as "nothing wrong". See that package's docstring.

export interface Availability {
    ok: boolean;
    reason: string;
    caveat: string;
}

export interface UnitSummary {
    name: string;
    load: string;
    active: string;
    sub: string;
    description: string;
}

export interface UnitList {
    available: Availability;
    scope: string;
    state_filter: string;
    units: UnitSummary[];
    truncated: boolean;
}

export interface Generation {
    number: number;
    date: string;
    nixos_version: string;
    kernel_version: string;
    configuration_revision: string;
    current: boolean;
}

export interface FilesystemUsage {
    mountpoint: string;
    device: string;
    fstype: string;
    total: number;
    used: number;
    free: number;
    percent: number;
}

export interface StorageReport {
    available: Availability;
    filesystems: { available: Availability; filesystems: FilesystemUsage[] };
    generations: { available: Availability; generations: Generation[] };
    nix_store: FilesystemUsage | null;
}

export interface ThrottleCounters {
    available: Availability;
    // Per PHYSICAL core (hyperthread siblings share one counter), not per
    // logical cpu - hence both cores_read and cpus_read.
    core_events: number;
    package_events: number;
    core_time_ms: number;
    package_time_ms: number;
    cpus_read: number;
    cores_read: number;
    cores_throttled: number;
}

export interface HostTemperature {
    chip: string;
    label: string;
    celsius: number;
    high: number | null;
    critical: number | null;
}

export interface ThermalReport {
    available: Availability;
    temperatures: HostTemperature[];
    throttling: ThrottleCounters;
    battery: {
        available: Availability;
        present: boolean;
        percent: number | null;
    };
    fans: { available: Availability; present: boolean };
}

export interface HostOverview {
    failed_system_units: UnitList;
    failed_user_units: UnitList;
    storage: StorageReport;
    thermal: ThermalReport;
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

// A human label for an auth mode wire value (the subscription login differs per
// backend). Null/empty -> a dash so a panel never shows a raw blank.
export function authLabel(mode: string | null): string {
    switch (mode) {
        case "chatgpt":
            return "ChatGPT";
        case "claude_ai":
            return "claude.ai";
        case "api_key":
            return "API key";
        case "local":
            return "Local";
        default:
            return mode || "-";
    }
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
    quota: UsageQuota | null;
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
    // The in-flight turn's prompt (steering stripped), present only while the run
    // is live, so a mid-turn reattach can render the user bubble the transcript
    // has not caught up on yet. null/absent when idle.
    prompt?: string | null;
}

// Friendly display names for the backend ids, shown on the cards. The list of
// SELECTABLE backends (and each one's default model) now comes from the server
// (GET /api/agents/backends, see BackendOption); these labels are the display
// fallback used when only a backend id is in hand (e.g. an agent card).
// "mock" is included so a dev-flag agent still reads cleanly.
export const BACKEND_LABELS: Record<string, string> = {
    codex: "Codex",
    claude: "Claude",
    opencode: "Opencode",
    mock: "Mock",
};

// The human label for a backend id, falling back to the raw id for anything
// unexpected (a legacy value that slipped through, say).
export function backendLabel(backend: string): string {
    return BACKEND_LABELS[backend] ?? backend;
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

// The reserved orchestrator agent's id: it is undeletable and configured from
// settings, so the UI hides its delete + settings affordances.
export const ORCHESTRATOR_ID = "orchestrator";

export const DEFAULT_POLL_SECONDS = 2;
// The host overview's own, much slower clock (it runs subprocesses server-side).
export const DEFAULT_HOST_OVERVIEW_SECONDS = 30;

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

// --- authentication ---------------------------------------------------------
//
// One seam for every call to the API, mirroring the backend's single enforcement
// middleware: it attaches the CSRF header the server requires on state-changing
// requests, and turns a 401 into a trip to the login page. Call `apiFetch`, never
// bare `fetch`, so a new call site cannot silently miss either.

export const CSRF_COOKIE = "scufris_csrf";
export const CSRF_HEADER = "X-Scufris-CSRF";

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

// The CSRF cookie is deliberately readable here (the session cookie is not):
// echoing it back in a header is what a cross-site attacker cannot do.
export function csrfToken(): string {
    const match = new RegExp(`(?:^|;\\s*)${CSRF_COOKIE}=([^;]*)`).exec(
        document.cookie,
    );
    return match ? decodeURIComponent(match[1]) : "";
}

function goToLogin(): void {
    // Never bounce a login page to itself: a 401 answered while ON /login/ means
    // "not logged in yet", which is exactly where the operator already is.
    if (window.location.pathname.startsWith("/login")) return;
    const next = window.location.pathname + window.location.search;
    window.location.assign(`/login/?next=${encodeURIComponent(next)}`);
}

export async function apiFetch(
    url: string,
    init: RequestInit = {},
): Promise<Response> {
    const method = (init.method ?? "GET").toUpperCase();
    const headers = new Headers(init.headers);
    if (!SAFE_METHODS.has(method)) {
        const token = csrfToken();
        if (token) headers.set(CSRF_HEADER, token);
    }
    const resp = await fetch(url, {
        ...init,
        method: init.method,
        headers,
        credentials: "same-origin",
    });
    if (resp.status === 401) goToLogin();
    return resp;
}

export async function logout(): Promise<void> {
    await apiFetch("/api/auth/logout", { method: "POST" });
    window.location.assign("/login/");
}

export async function fetchJson<T>(url: string): Promise<T> {
    const resp = await apiFetch(url);
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
    const resp = await apiFetch(url, {
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
        return {
            poll_seconds: DEFAULT_POLL_SECONDS,
            agent_enabled: false,
            host_overview_seconds: DEFAULT_HOST_OVERVIEW_SECONDS,
        };
    }
}
