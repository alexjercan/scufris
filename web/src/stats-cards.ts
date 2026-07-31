// The live gauges from /api/stats: one card per subsystem (cpu, load, gpu,
// memory, disks, network). Each is handed a pre-built sparkline by
// `stats-view.ts`, which owns the poll and the history keys.

import { el, escapeHtml, formatBytes } from "./common";
import { type DiskIoRate, type GpuStats, type HostStats } from "./stats-types";
import {
    bar,
    card,
    formatUptime,
    severity,
    tempSeverity,
} from "./stats-elements";

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

export function cpuCard(stats: HostStats, spark: HTMLElement): HTMLElement {
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

export function memoryCard(stats: HostStats, spark: HTMLElement): HTMLElement {
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

export function loadCard(stats: HostStats, spark: HTMLElement): HTMLElement {
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

export function baseDisks(stats: HostStats): DiskIoRate[] {
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

export function disksCard(stats: HostStats, spark: HTMLElement): HTMLElement {
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

export function networkCard(stats: HostStats, spark: HTMLElement): HTMLElement {
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

export function gpuCard(gpu: GpuStats, spark: HTMLElement): HTMLElement {
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
