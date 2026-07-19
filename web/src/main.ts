import "./style.css";

// Mirrors scufris.metrics.HostStats (the /api/stats payload).
interface MemStats {
    total: number;
    used: number;
    available: number;
    percent: number;
}

interface SwapStats {
    total: number;
    used: number;
    percent: number;
}

interface DiskUsage {
    mountpoint: string;
    total: number;
    used: number;
    percent: number;
}

interface NetIO {
    bytes_sent: number;
    bytes_recv: number;
}

interface HostStats {
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

interface AppConfig {
    poll_seconds: number;
}

const DEFAULT_POLL_SECONDS = 2;

function formatBytes(bytes: number): string {
    if (bytes <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB", "PB"];
    const exp = Math.min(
        units.length - 1,
        Math.floor(Math.log(bytes) / Math.log(1024)),
    );
    const value = bytes / Math.pow(1024, exp);
    return `${value.toFixed(exp === 0 ? 0 : 1)} ${units[exp]}`;
}

function formatUptime(seconds: number): string {
    const total = Math.floor(seconds);
    const days = Math.floor(total / 86400);
    const hours = Math.floor((total % 86400) / 3600);
    const mins = Math.floor((total % 3600) / 60);
    const parts: string[] = [];
    if (days > 0) parts.push(`${days}d`);
    if (hours > 0 || days > 0) parts.push(`${hours}h`);
    parts.push(`${mins}m`);
    return parts.join(" ");
}

function severity(percent: number): string {
    if (percent >= 90) return "is-crit";
    if (percent >= 75) return "is-warn";
    return "";
}

function el(tag: string, className?: string, html?: string): HTMLElement {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (html !== undefined) node.innerHTML = html;
    return node;
}

function bar(percent: number): HTMLElement {
    const wrap = el("div", "bar");
    const fill = el("div", `bar__fill ${severity(percent)}`.trim());
    fill.style.width = `${Math.max(0, Math.min(100, percent)).toFixed(1)}%`;
    wrap.appendChild(fill);
    return wrap;
}

function card(
    title: string,
    extra: string,
    body: (root: HTMLElement) => void,
): HTMLElement {
    const root = el("section", "card");
    root.appendChild(
        el("h2", "card__title", `<span>${title}</span><span>${extra}</span>`),
    );
    body(root);
    return root;
}

function cpuCard(stats: HostStats): HTMLElement {
    return card("CPU", `${stats.per_cpu_percent.length} cores`, (root) => {
        root.appendChild(
            el(
                "div",
                "card__value",
                `${stats.cpu_percent.toFixed(1)}<small>%</small>`,
            ),
        );
        root.appendChild(bar(stats.cpu_percent));
        const cores = el("div", "cores");
        for (const pct of stats.per_cpu_percent) {
            const core = el("div", "core");
            const fill = el("div", `core__fill ${severity(pct)}`.trim());
            fill.style.height = `${Math.max(0, Math.min(100, pct)).toFixed(0)}%`;
            core.appendChild(fill);
            core.title = `${pct.toFixed(0)}%`;
            cores.appendChild(core);
        }
        root.appendChild(cores);
    });
}

function usageCard(
    title: string,
    used: number,
    total: number,
    percent: number,
): HTMLElement {
    return card(title, formatBytes(total), (root) => {
        root.appendChild(
            el("div", "card__value", `${percent.toFixed(1)}<small>%</small>`),
        );
        root.appendChild(bar(percent));
        const rows = el("div", "card__rows");
        rows.appendChild(
            el(
                "div",
                "row",
                `<span>used</span><span>${formatBytes(used)}</span>`,
            ),
        );
        rows.appendChild(
            el(
                "div",
                "row",
                `<span>free</span><span>${formatBytes(Math.max(0, total - used))}</span>`,
            ),
        );
        root.appendChild(rows);
    });
}

function loadCard(stats: HostStats): HTMLElement {
    const [one, five, fifteen] = stats.load_avg;
    return card("Load average", "1 / 5 / 15 min", (root) => {
        root.appendChild(
            el(
                "div",
                "card__value",
                `${one.toFixed(2)}<small>${five.toFixed(2)} ${fifteen.toFixed(2)}</small>`,
            ),
        );
    });
}

function disksCard(stats: HostStats): HTMLElement {
    return card("Disks", `${stats.disks.length} mounts`, (root) => {
        const rows = el("div", "card__rows");
        for (const disk of stats.disks) {
            rows.appendChild(
                el(
                    "div",
                    "row",
                    `<span>${disk.mountpoint}</span><span>${disk.percent.toFixed(0)}% of ${formatBytes(disk.total)}</span>`,
                ),
            );
            rows.appendChild(bar(disk.percent));
        }
        if (stats.disks.length === 0) {
            rows.appendChild(
                el("div", "row", "<span>no mounts</span><span></span>"),
            );
        }
        root.appendChild(rows);
    });
}

function netCard(stats: HostStats): HTMLElement {
    return card("Network", "since boot", (root) => {
        const rows = el("div", "card__rows");
        rows.appendChild(
            el(
                "div",
                "row",
                `<span>received</span><span>${formatBytes(stats.net.bytes_recv)}</span>`,
            ),
        );
        rows.appendChild(
            el(
                "div",
                "row",
                `<span>sent</span><span>${formatBytes(stats.net.bytes_sent)}</span>`,
            ),
        );
        root.appendChild(rows);
    });
}

function renderSummary(stats: HostStats): void {
    const summary = document.getElementById("host-summary");
    if (!summary) return;
    summary.innerHTML = "";
    const bits: [string, string][] = [
        ["host", stats.hostname],
        ["os", `${stats.os_name} ${stats.kernel}`],
        ["up", formatUptime(stats.uptime_seconds)],
    ];
    for (const [label, value] of bits) {
        summary.appendChild(
            el("span", "", `${label} <strong>${value}</strong>`),
        );
    }
}

function renderCards(stats: HostStats): void {
    const cards = document.getElementById("cards");
    if (!cards) return;
    cards.replaceChildren(
        cpuCard(stats),
        usageCard("Memory", stats.mem.used, stats.mem.total, stats.mem.percent),
        usageCard(
            "Swap",
            stats.swap.used,
            stats.swap.total,
            stats.swap.percent,
        ),
        loadCard(stats),
        disksCard(stats),
        netCard(stats),
    );
}

function setStatus(text: string, isError = false): void {
    const status = document.getElementById("status");
    if (!status) return;
    status.textContent = text;
    status.classList.toggle("is-error", isError);
}

async function fetchJson<T>(url: string): Promise<T> {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`${url} -> ${String(resp.status)}`);
    return (await resp.json()) as T;
}

async function refresh(): Promise<void> {
    const stats = await fetchJson<HostStats>("/api/stats");
    renderSummary(stats);
    renderCards(stats);
    const time = new Date(stats.sampled_at).toLocaleTimeString();
    setStatus(`updated ${time}`);
}

async function loadPollSeconds(): Promise<number> {
    try {
        const config = await fetchJson<AppConfig>("/api/config");
        return config.poll_seconds > 0
            ? config.poll_seconds
            : DEFAULT_POLL_SECONDS;
    } catch {
        return DEFAULT_POLL_SECONDS;
    }
}

async function main(): Promise<void> {
    const pollSeconds = await loadPollSeconds();
    const tick = (): void => {
        refresh().catch((err: unknown) => {
            console.error(err);
            setStatus("cannot reach backend - retrying", true);
        });
    };
    tick();
    setInterval(tick, pollSeconds * 1000);
}

void main();
