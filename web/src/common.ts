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

export interface AgentInfo {
    model: string;
    auth_mode: string;
    enabled: boolean;
}

export interface AgentTool {
    name: string;
    description: string;
}

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
