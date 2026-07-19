// Dashboard logic: rendering + polling + chat. Kept free of import-time side
// effects (no auto-start, no CSS import) so the render functions are importable
// by the jsdom tests. `main.ts` is the thin entry that imports the styles and
// calls `start()`.

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

interface AppConfig {
    poll_seconds: number;
    agent_enabled: boolean;
}

interface ChatReply {
    text: string;
    status?: string;
}

const DEFAULT_POLL_SECONDS = 2;

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
                    `<span>${escapeHtml(disk.mountpoint)}</span><span>${disk.percent.toFixed(0)}% of ${formatBytes(disk.total)}</span>`,
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

export function renderSummary(stats: HostStats): void {
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
            el("span", "", `${label} <strong>${escapeHtml(value)}</strong>`),
        );
    }
}

export function renderCards(stats: HostStats): void {
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

async function loadConfig(): Promise<AppConfig> {
    try {
        return await fetchJson<AppConfig>("/api/config");
    } catch {
        return { poll_seconds: DEFAULT_POLL_SECONDS, agent_enabled: false };
    }
}

// --- Chat panel ---------------------------------------------------------

function appendMessage(
    log: HTMLElement,
    role: string,
    text: string,
): HTMLElement {
    const msg = el("div", `chat__msg chat__msg--${role}`);
    msg.textContent = text;
    log.appendChild(msg);
    log.scrollTop = log.scrollHeight;
    return msg;
}

async function sendChat(message: string): Promise<ChatReply> {
    const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
    });
    if (!resp.ok) {
        const detail = (await resp.json().catch(() => null)) as {
            detail?: string;
        } | null;
        throw new Error(
            detail?.detail || `chat failed (${String(resp.status)})`,
        );
    }
    return (await resp.json()) as ChatReply;
}

function initChat(config: AppConfig): void {
    const form = document.getElementById("chat-form") as HTMLFormElement | null;
    const input = document.getElementById(
        "chat-input",
    ) as HTMLInputElement | null;
    const log = document.getElementById("chat-log");
    const reset = document.getElementById("chat-reset");
    if (!form || !input || !log || !reset) return;

    if (!config.agent_enabled) {
        appendMessage(
            log,
            "system",
            "agent is disabled. Set SCUFRIS_AGENT_ENABLED=1 and run `codex login`.",
        );
        input.disabled = true;
        return;
    }

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const message = input.value.trim();
        if (!message) return;
        appendMessage(log, "user", message);
        input.value = "";
        input.disabled = true;
        const pending = appendMessage(log, "assistant", "...");
        sendChat(message)
            .then((reply) => {
                pending.textContent = reply.text || "(no reply)";
            })
            .catch((err: unknown) => {
                pending.classList.add("chat__msg--error");
                pending.textContent =
                    err instanceof Error ? err.message : "error";
            })
            .finally(() => {
                input.disabled = false;
                input.focus();
                log.scrollTop = log.scrollHeight;
            });
    });

    reset.addEventListener("click", () => {
        void fetch("/api/chat/reset", { method: "POST" }).finally(() => {
            log.replaceChildren();
        });
    });
}

export async function start(): Promise<void> {
    const config = await loadConfig();
    initChat(config);
    const pollSeconds =
        config.poll_seconds > 0 ? config.poll_seconds : DEFAULT_POLL_SECONDS;
    const tick = (): void => {
        refresh().catch((err: unknown) => {
            console.error(err);
            setStatus("cannot reach backend - retrying", true);
        });
    };
    tick();
    setInterval(tick, pollSeconds * 1000);
}
