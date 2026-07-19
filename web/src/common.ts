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
}

export interface AppConfig {
    poll_seconds: number;
    agent_enabled: boolean;
}

export interface ChatReply {
    text: string;
    status?: string;
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

export async function loadConfig(): Promise<AppConfig> {
    try {
        return await fetchJson<AppConfig>("/api/config");
    } catch {
        return { poll_seconds: DEFAULT_POLL_SECONDS, agent_enabled: false };
    }
}
