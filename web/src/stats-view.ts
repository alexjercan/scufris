// Stats page: render the host-metrics dashboard and poll it. No import-time
// side effects (the `stats.ts` entry calls `startStats`), so the render
// functions are importable by the jsdom tests.

import {
    DEFAULT_POLL_SECONDS,
    el,
    escapeHtml,
    fetchJson,
    loadConfig,
    type GpuStats,
    type HostStats,
} from "./common";

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

function row(label: string, value: string): HTMLElement {
    return el("div", "row", `<span>${label}</span><span>${value}</span>`);
}

function rate(bytesPerSec: number): string {
    return `${formatBytes(bytesPerSec)}/s`;
}

function gpuCard(gpu: GpuStats): HTMLElement {
    return card("GPU", escapeHtml(gpu.name), (root) => {
        root.appendChild(
            el(
                "div",
                "card__value",
                `${gpu.util_percent.toFixed(0)}<small>%</small>`,
            ),
        );
        root.appendChild(bar(gpu.util_percent));
        root.appendChild(bar(gpu.mem_percent));
        const rows2 = el("div", "card__rows");
        rows2.appendChild(
            row(
                "vram",
                `${formatBytes(gpu.mem_used_mb * 1048576)} / ${formatBytes(gpu.mem_total_mb * 1048576)}`,
            ),
        );
        rows2.appendChild(row("temp", `${gpu.temp_c.toFixed(0)} C`));
        rows2.appendChild(
            row(
                "power",
                `${gpu.power_w.toFixed(0)} / ${gpu.power_limit_w.toFixed(0)} W`,
            ),
        );
        rows2.appendChild(
            row("clocks", `${gpu.clock_sm_mhz} / ${gpu.clock_mem_mhz} MHz`),
        );
        root.appendChild(rows2);
    });
}

function freqCard(stats: HostStats): HTMLElement | null {
    const freqs = stats.per_cpu_freq_mhz;
    if (freqs.length === 0) return null;
    const avg = freqs.reduce((a, b) => a + b, 0) / freqs.length;
    const max = Math.max(...freqs);
    const min = Math.min(...freqs);
    return card("CPU frequency", `${freqs.length} cores`, (root) => {
        root.appendChild(
            el(
                "div",
                "card__value",
                `${(avg / 1000).toFixed(2)}<small>GHz avg</small>`,
            ),
        );
        const rows = el("div", "card__rows");
        rows.appendChild(row("min", `${(min / 1000).toFixed(2)} GHz`));
        rows.appendChild(row("max", `${(max / 1000).toFixed(2)} GHz`));
        root.appendChild(rows);
    });
}

function sensorsCard(stats: HostStats): HTMLElement | null {
    const groups = stats.temps;
    if (groups.length === 0) return null;
    const count = groups.reduce((n, g) => n + g.readings.length, 0);
    return card("Temperatures", `${count} sensors`, (root) => {
        const rows = el("div", "card__rows");
        for (const group of groups) {
            for (const reading of group.readings) {
                const hot =
                    reading.high !== null && reading.current >= reading.high;
                const rowEl = row(
                    `${escapeHtml(group.chip)} ${escapeHtml(reading.label)}`,
                    `${reading.current.toFixed(0)} C`,
                );
                if (hot) rowEl.classList.add("is-hot");
                rows.appendChild(rowEl);
            }
        }
        root.appendChild(rows);
    });
}

function netIfCard(stats: HostStats): HTMLElement | null {
    const nics = stats.net_interfaces.filter(
        (n) => n.sent_per_sec > 0 || n.recv_per_sec > 0,
    );
    if (nics.length === 0) return null;
    return card("Network interfaces", `${nics.length} active`, (root) => {
        const rows = el("div", "card__rows");
        for (const nic of nics.slice(0, 6)) {
            rows.appendChild(
                row(
                    escapeHtml(nic.name),
                    `down ${rate(nic.recv_per_sec)} / up ${rate(nic.sent_per_sec)}`,
                ),
            );
        }
        root.appendChild(rows);
    });
}

function diskIoCard(stats: HostStats): HTMLElement | null {
    const disks = stats.disk_io.filter(
        (d) => d.read_per_sec > 0 || d.write_per_sec > 0,
    );
    if (disks.length === 0) return null;
    return card("Disk IO", `${disks.length} active`, (root) => {
        const rows = el("div", "card__rows");
        for (const disk of disks.slice(0, 6)) {
            rows.appendChild(
                row(
                    escapeHtml(disk.name),
                    `read ${rate(disk.read_per_sec)} / write ${rate(disk.write_per_sec)}`,
                ),
            );
        }
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
    const items: (HTMLElement | null)[] = [
        cpuCard(stats),
        freqCard(stats),
        ...stats.gpus.map(gpuCard),
        usageCard("Memory", stats.mem.used, stats.mem.total, stats.mem.percent),
        usageCard(
            "Swap",
            stats.swap.used,
            stats.swap.total,
            stats.swap.percent,
        ),
        sensorsCard(stats),
        loadCard(stats),
        disksCard(stats),
        diskIoCard(stats),
        netCard(stats),
        netIfCard(stats),
    ];
    cards.replaceChildren(...items.filter((c): c is HTMLElement => c !== null));
}

function setStatus(text: string, isError = false): void {
    const status = document.getElementById("status");
    if (!status) return;
    status.textContent = text;
    status.classList.toggle("is-error", isError);
}

async function refresh(): Promise<void> {
    const stats = await fetchJson<HostStats>("/api/stats");
    renderSummary(stats);
    renderCards(stats);
    const time = new Date(stats.sampled_at).toLocaleTimeString();
    setStatus(`updated ${time}`);
}

export async function startStats(): Promise<void> {
    const config = await loadConfig();
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
