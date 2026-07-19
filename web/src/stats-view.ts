// Stats page: render the host-metrics dashboard and poll it. No import-time
// side effects (the `stats.ts` entry calls `startStats`), so the render
// functions are importable by the jsdom tests.

import {
    DEFAULT_POLL_SECONDS,
    el,
    escapeHtml,
    fetchJson,
    formatBytes,
    loadConfig,
    type DiskIoRate,
    type GpuStats,
    type HostStats,
} from "./common";

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

const SVG_NS = "http://www.w3.org/2000/svg";

// Client-side rolling history: each poll appends one sample per series into a
// bounded array, so the sparklines fill from page-load the way btop's graphs
// fill from start (no backend history needed - /api/stats already carries every
// value graphed). Keyed by series id (cpu, load, mem, disk, net, gpu:<i>).
// Module state, so a test-reset hook is exported (see the lesson
// persistent-ui-state-needs-a-test-reset-hook).
const HISTORY_LEN = 60;
const _history = new Map<string, number[]>();

function push(key: string, value: number): number[] {
    const arr = _history.get(key) ?? [];
    arr.push(value);
    while (arr.length > HISTORY_LEN) arr.shift();
    _history.set(key, arr);
    return arr;
}

export function _resetStatsHistory(): void {
    _history.clear();
}

// A btop-style mini graph: a filled area under a polyline, drawn as inline SVG
// on a fixed 100x30 viewBox and stretched to the card width by CSS
// (preserveAspectRatio=none). Percent series pass max=100; rate/load series
// autoscale to the window max (floored at 1 to avoid divide-by-zero). Empty-safe
// and pure (holds no state), so it is unit-testable in jsdom.
export function sparkline(
    values: number[],
    max?: number,
    sevClass = "",
    title = "",
): SVGElement {
    const w = 100;
    const h = 30;
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", `spark ${sevClass}`.trim());
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svg.setAttribute("preserveAspectRatio", "none");
    // An SVG <title> is the native hover tooltip; it also names the graph for
    // assistive tech. Rendered even with no data so an empty graph still hints.
    if (title) {
        const t = document.createElementNS(SVG_NS, "title");
        t.textContent = title;
        svg.appendChild(t);
    }
    if (values.length === 0) return svg;
    const ceil = Math.max(max ?? Math.max(...values), 1);
    const n = values.length;
    const xAt = (i: number): number => (n === 1 ? w : (i / (n - 1)) * w);
    const yAt = (v: number): number =>
        h - (Math.max(0, Math.min(ceil, v)) / ceil) * h;
    const pts = values.map(
        (v, i) => `${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`,
    );
    const area = document.createElementNS(SVG_NS, "polygon");
    area.setAttribute("class", "spark__area");
    area.setAttribute("points", `0,${h} ${pts.join(" ")} ${w},${h}`);
    svg.appendChild(area);
    const line = document.createElementNS(SVG_NS, "polyline");
    line.setAttribute("class", "spark__line");
    line.setAttribute("points", pts.join(" "));
    svg.appendChild(line);
    return svg;
}

// A sparkline with a btop-style corner caption. `label` is the short visible
// tag (e.g. "cpu %"); `title` is the fuller hover tooltip. Returns a positioned
// wrapper so the caption floats over the graph without changing its height.
function labeledSpark(
    label: string,
    title: string,
    values: number[],
    max?: number,
    sevClass = "",
): HTMLElement {
    const wrap = el("div", "spark-wrap");
    const caption = el("span", "spark__label");
    caption.textContent = label;
    wrap.appendChild(caption);
    wrap.appendChild(sparkline(values, max, sevClass, title));
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

function cpuCard(stats: HostStats, spark: HTMLElement): HTMLElement {
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
        root.appendChild(spark);
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

function memoryCard(stats: HostStats, spark: HTMLElement): HTMLElement {
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
        root.appendChild(spark);
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

function perSec(n: number): string {
    if (n >= 1000) return `${(n / 1000).toFixed(1)}k/s`;
    return `${Math.round(n)}/s`;
}

function loadCard(stats: HostStats, spark: HTMLElement): HTMLElement {
    const [one, five, fifteen] = stats.load_avg;
    return card("Load average", "1 / 5 / 15 min", (root) => {
        root.appendChild(
            el(
                "div",
                "card__value",
                `${one.toFixed(2)}<small>${five.toFixed(2)} ${fifteen.toFixed(2)}</small>`,
            ),
        );
        root.appendChild(spark);
        const rows = el("div", "card__rows");
        rows.appendChild(row("tasks", `${stats.process_count}`));
        rows.appendChild(
            row(
                "ctx switches",
                perSec(stats.cpu_activity.ctx_switches_per_sec),
            ),
        );
        rows.appendChild(
            row("interrupts", perSec(stats.cpu_activity.interrupts_per_sec)),
        );
        rows.appendChild(row("uptime", formatUptime(stats.uptime_seconds)));
        root.appendChild(rows);
    });
}

// Base physical disks only: drop loop/ram/dm/sr noise and partitions (a device
// whose name has another device as a strict prefix, e.g. nvme0n1p1 under
// nvme0n1). Sorted for a stable row order across polls.
const _DISK_NOISE = /^(loop|ram|dm-|sr|zram|fd)/;

function baseDisks(stats: HostStats): DiskIoRate[] {
    const names = stats.disk_io.map((d) => d.name);
    return stats.disk_io
        .filter((d) => !_DISK_NOISE.test(d.name))
        .filter((d) => !names.some((o) => o !== d.name && d.name.startsWith(o)))
        .sort((a, b) => a.name.localeCompare(b.name));
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

function disksCard(stats: HostStats, spark: HTMLElement): HTMLElement {
    return card("Disks", `${stats.disks.length} mounts`, (root) => {
        root.appendChild(spark);
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

        // Always render the base disks (stable row set), with a dash when idle,
        // so the card does not resize as IO blinks in and out.
        const io = baseDisks(stats);
        if (io.length > 0) {
            root.appendChild(el("div", "card__subhead", "io"));
            const iorows = el("div", "card__rows");
            for (const d of io) {
                const idle = d.read_per_sec === 0 && d.write_per_sec === 0;
                iorows.appendChild(
                    row(
                        escapeHtml(d.name),
                        idle
                            ? "-"
                            : `r ${rate(d.read_per_sec)} / w ${rate(d.write_per_sec)}`,
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

function networkCard(stats: HostStats, spark: HTMLElement): HTMLElement {
    const nics = stats.net_interfaces
        .filter((n) => n.sent_per_sec > 0 || n.recv_per_sec > 0)
        .slice(0, 6);
    return card("Network", "live + since boot", (root) => {
        root.appendChild(spark);
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

function gpuCard(gpu: GpuStats, spark: HTMLElement): HTMLElement {
    return card("GPU", escapeHtml(gpu.name), (root) => {
        root.appendChild(
            el(
                "div",
                "card__value",
                `${gpu.util_percent.toFixed(0)}<small>%</small>`,
            ),
        );
        root.appendChild(bar(gpu.util_percent));
        root.appendChild(spark);
        const rows2 = el("div", "card__rows");
        rows2.appendChild(
            row(
                "vram",
                `${formatBytes(gpu.mem_used_mb * 1048576)} / ${formatBytes(gpu.mem_total_mb * 1048576)}`,
            ),
        );
        // VRAM fill bar sits directly under its numbers (not above), so the bar
        // reads as belonging to the vram row.
        rows2.appendChild(bar(gpu.mem_percent));
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

// Aggregate rates for the disk/network sparklines: total bytes/s across the
// real base disks and across all interfaces, so each card graphs one honest
// throughput line rather than one line per device.
function totalDiskIo(stats: HostStats): number {
    return baseDisks(stats).reduce(
        (sum, d) => sum + d.read_per_sec + d.write_per_sec,
        0,
    );
}

function totalNetIo(stats: HostStats): number {
    return stats.net_interfaces.reduce(
        (sum, n) => sum + n.sent_per_sec + n.recv_per_sec,
        0,
    );
}

export function renderCards(stats: HostStats): void {
    const cards = document.getElementById("cards");
    if (!cards) return;
    // Push this poll's sample into each series first, then graph the window.
    // Percent series clamp to 0-100 (max=100) and colour by the latest value's
    // severity; rate/load series autoscale to their own window (neutral). Each
    // graph carries a short corner label and a fuller hover tooltip.
    const cpuSpark = labeledSpark(
        "cpu %",
        "CPU utilization (%)",
        push("cpu", stats.cpu_percent),
        100,
        severity(stats.cpu_percent),
    );
    const loadSpark = labeledSpark(
        "load 1m",
        "Load average (1 min)",
        push("load", stats.load_avg[0]),
    );
    const memSpark = labeledSpark(
        "mem %",
        "Memory used (%)",
        push("mem", stats.mem.percent),
        100,
        severity(stats.mem.percent),
    );
    const diskSpark = labeledSpark(
        "disk i/o",
        "Disk I/O (read+write, bytes/s)",
        push("disk", totalDiskIo(stats)),
    );
    const netSpark = labeledSpark(
        "net i/o",
        "Network (up+down, bytes/s)",
        push("net", totalNetIo(stats)),
    );
    const items: HTMLElement[] = [
        cpuCard(stats, cpuSpark),
        loadCard(stats, loadSpark),
        ...stats.gpus.map((g, i) =>
            gpuCard(
                g,
                labeledSpark(
                    "gpu %",
                    "GPU utilization (%)",
                    push(`gpu:${i}`, g.util_percent),
                    100,
                    severity(g.util_percent),
                ),
            ),
        ),
        memoryCard(stats, memSpark),
        disksCard(stats, diskSpark),
        networkCard(stats, netSpark),
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
