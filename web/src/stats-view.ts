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

function tempSeverity(temp: number): string {
    if (temp >= 85) return "is-crit";
    if (temp >= 70) return "is-warn";
    return "";
}

function coretempGroup(stats: HostStats) {
    return stats.temps.find((g) => g.chip === "coretemp");
}

// Physical-core temperatures ("Core N"), in order. There are usually fewer of
// these than logical CPUs (hyperthreading), so cpuCard maps them across the
// load squares by index proportion - an approximation, not a 1:1 core mapping.
function cpuCoreTemps(stats: HostStats): number[] {
    const group = coretempGroup(stats);
    if (!group) return [];
    return group.readings
        .filter((r) => r.label.toLowerCase().startsWith("core"))
        .map((r) => r.current);
}

function cpuPackageTemp(stats: HostStats): number | null {
    const pkg = coretempGroup(stats)?.readings.find((r) =>
        r.label.toLowerCase().startsWith("package"),
    );
    return pkg ? pkg.current : null;
}

function cpuCard(stats: HostStats): HTMLElement {
    const coreTemps = cpuCoreTemps(stats);
    const n = stats.per_cpu_percent.length;
    return card("CPU", `${n} cores`, (root) => {
        root.appendChild(
            el(
                "div",
                "card__value",
                `${stats.cpu_percent.toFixed(1)}<small>%</small>`,
            ),
        );
        root.appendChild(bar(stats.cpu_percent));
        const cores = el("div", "cores");
        stats.per_cpu_percent.forEach((pct, i) => {
            const core = el("div", "core");
            const fill = el("div", `core__fill ${severity(pct)}`.trim());
            fill.style.height = `${Math.max(0, Math.min(100, pct)).toFixed(0)}%`;
            core.appendChild(fill);
            let title = `${pct.toFixed(0)}%`;
            if (coreTemps.length > 0) {
                const t =
                    coreTemps[
                        Math.min(
                            coreTemps.length - 1,
                            Math.floor((i * coreTemps.length) / n),
                        )
                    ];
                const label = el(
                    "span",
                    `core__temp ${tempSeverity(t)}`.trim(),
                );
                label.textContent = t.toFixed(0);
                core.appendChild(label);
                title += ` / ${t.toFixed(0)} C`;
            }
            core.title = title;
            cores.appendChild(core);
        });
        root.appendChild(cores);
        const rows = el("div", "card__rows");
        const freqs = stats.per_cpu_freq_mhz;
        if (freqs.length > 0) {
            const avg = freqs.reduce((a, b) => a + b, 0) / freqs.length;
            rows.appendChild(row("freq", `${(avg / 1000).toFixed(2)} GHz avg`));
        }
        const pkg = cpuPackageTemp(stats);
        if (pkg !== null)
            rows.appendChild(row("package", `${pkg.toFixed(0)} C`));
        if (rows.childElementCount > 0) root.appendChild(rows);
    });
}

function memoryCard(stats: HostStats): HTMLElement {
    const mem = stats.mem;
    const swap = stats.swap;
    return card("Memory", formatBytes(mem.total), (root) => {
        root.appendChild(
            el(
                "div",
                "card__value",
                `${mem.percent.toFixed(1)}<small>%</small>`,
            ),
        );
        root.appendChild(bar(mem.percent));
        const rows = el("div", "card__rows");
        rows.appendChild(
            row("used", `${formatBytes(mem.used)} / ${formatBytes(mem.total)}`),
        );
        rows.appendChild(row("available", formatBytes(mem.available)));
        root.appendChild(rows);
        if (swap.total > 0) {
            root.appendChild(el("div", "card__subhead", "swap"));
            root.appendChild(bar(swap.percent));
            const srows = el("div", "card__rows");
            srows.appendChild(
                row(
                    "used",
                    `${formatBytes(swap.used)} / ${formatBytes(swap.total)} (${swap.percent.toFixed(0)}%)`,
                ),
            );
            root.appendChild(srows);
        }
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

// Disk temperatures live in nvme/drivetemp-style chips (not coretemp/acpitz).
function diskTempReadings(
    stats: HostStats,
): { label: string; current: number }[] {
    const out: { label: string; current: number }[] = [];
    for (const group of stats.temps) {
        const chip = group.chip.toLowerCase();
        if (
            chip.includes("nvme") ||
            chip.includes("drivetemp") ||
            chip.includes("disk")
        ) {
            for (const r of group.readings) {
                out.push({
                    label: `${group.chip} ${r.label}`,
                    current: r.current,
                });
            }
        }
    }
    return out;
}

function disksCard(stats: HostStats): HTMLElement {
    return card("Disks", `${stats.disks.length} mounts`, (root) => {
        const rows = el("div", "card__rows");
        for (const disk of stats.disks) {
            rows.appendChild(
                row(
                    escapeHtml(disk.mountpoint),
                    `${disk.percent.toFixed(0)}% of ${formatBytes(disk.total)}`,
                ),
            );
            rows.appendChild(bar(disk.percent));
        }
        if (stats.disks.length === 0) rows.appendChild(row("no mounts", ""));
        root.appendChild(rows);

        const io = stats.disk_io
            .filter((d) => d.read_per_sec > 0 || d.write_per_sec > 0)
            .slice(0, 6);
        if (io.length > 0) {
            root.appendChild(el("div", "card__subhead", "io"));
            const iorows = el("div", "card__rows");
            for (const d of io) {
                iorows.appendChild(
                    row(
                        escapeHtml(d.name),
                        `r ${rate(d.read_per_sec)} / w ${rate(d.write_per_sec)}`,
                    ),
                );
            }
            root.appendChild(iorows);
        }

        const temps = diskTempReadings(stats);
        if (temps.length > 0) {
            root.appendChild(el("div", "card__subhead", "temp"));
            const trows = el("div", "card__rows");
            for (const t of temps) {
                const r = row(escapeHtml(t.label), `${t.current.toFixed(0)} C`);
                if (t.current >= 60) r.classList.add("is-hot");
                trows.appendChild(r);
            }
            root.appendChild(trows);
        }
    });
}

function networkCard(stats: HostStats): HTMLElement {
    const nics = stats.net_interfaces
        .filter((n) => n.sent_per_sec > 0 || n.recv_per_sec > 0)
        .slice(0, 6);
    return card("Network", "live + since boot", (root) => {
        const rows = el("div", "card__rows");
        if (nics.length > 0) {
            for (const nic of nics) {
                rows.appendChild(
                    row(
                        escapeHtml(nic.name),
                        `down ${rate(nic.recv_per_sec)} / up ${rate(nic.sent_per_sec)}`,
                    ),
                );
            }
        } else {
            rows.appendChild(row("idle", ""));
        }
        root.appendChild(rows);
        root.appendChild(el("div", "card__subhead", "since boot"));
        const totals = el("div", "card__rows");
        totals.appendChild(row("received", formatBytes(stats.net.bytes_recv)));
        totals.appendChild(row("sent", formatBytes(stats.net.bytes_sent)));
        root.appendChild(totals);
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
    const items: HTMLElement[] = [
        cpuCard(stats),
        loadCard(stats),
        ...stats.gpus.map(gpuCard),
        memoryCard(stats),
        disksCard(stats),
        networkCard(stats),
    ];
    cards.replaceChildren(...items);
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
