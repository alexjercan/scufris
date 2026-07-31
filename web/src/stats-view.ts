// Stats page: render the host-metrics dashboard and poll it. No import-time
// side effects (the `stats.ts` entry calls `startStats`), so the render
// functions are importable by the jsdom tests.
//
// The cards live beside this file: the live gauges in `stats-cards.ts`, the
// host-inspection cards in `stats-host-cards.ts`, and the pieces both read in
// `stats-elements.ts`.

import {
    DEFAULT_HOST_OVERVIEW_SECONDS,
    DEFAULT_POLL_SECONDS,
    el,
    escapeHtml,
    fetchJson,
    loadConfig,
} from "./common";
import { type HostOverview, type HostStats } from "./stats-types";
import {
    formatUptime,
    labeledSpark,
    pushHistory,
    severity,
} from "./stats-elements";
import {
    baseDisks,
    cpuCard,
    disksCard,
    gpuCard,
    loadCard,
    memoryCard,
    networkCard,
} from "./stats-cards";
import {
    failedUnitsCard,
    generationsCard,
    nixStoreCard,
    thermalCard,
} from "./stats-host-cards";

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
        pushHistory("cpu", stats.cpu_percent),
        100,
        severity(stats.cpu_percent),
    );
    const loadSpark = labeledSpark(
        "load 1m",
        "Load average (1 min)",
        pushHistory("load", stats.load_avg[0]),
    );
    const memSpark = labeledSpark(
        "mem %",
        "Memory used (%)",
        pushHistory("mem", stats.mem.percent),
        100,
        severity(stats.mem.percent),
    );
    const diskSpark = labeledSpark(
        "disk i/o",
        "Disk I/O (read+write, bytes/s)",
        pushHistory("disk", totalDiskIo(stats)),
    );
    const netSpark = labeledSpark(
        "net i/o",
        "Network (up+down, bytes/s)",
        pushHistory("net", totalNetIo(stats)),
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
                    pushHistory(`gpu:${i}`, g.util_percent),
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

export function renderHostCards(overview: HostOverview): void {
    const host = document.getElementById("host-cards");
    if (!host) return;
    host.replaceChildren(
        failedUnitsCard(overview),
        generationsCard(overview),
        nixStoreCard(overview),
        thermalCard(overview),
    );
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

async function refreshHost(): Promise<void> {
    renderHostCards(await fetchJson<HostOverview>("/api/host/overview"));
}

// A failed host poll must not leave the previous snapshot sitting there looking
// current: indefinitely-stale cards are the same lie as a blank one, just
// harder to notice. Stamp the section instead of silently keeping old data.
export function markHostCardsStale(detail: string): void {
    const host = document.getElementById("host-cards");
    if (!host) return;
    const existing = host.querySelector(".host-stale");
    const note = existing ?? el("div", "card__note is-error host-stale");
    note.textContent = `host inspection not updating: ${detail}`;
    if (!existing) host.appendChild(note);
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

    // The host overview on its own clock. A failure here does NOT touch the
    // status line: the gauges are still live, and reporting "cannot reach
    // backend" because a nixos-rebuild call was slow would be a lie about the
    // part of the page that is working.
    const hostSeconds =
        config.host_overview_seconds > 0
            ? config.host_overview_seconds
            : DEFAULT_HOST_OVERVIEW_SECONDS;
    const hostTick = (): void => {
        refreshHost().catch((err: unknown) => {
            console.error(err);
            markHostCardsStale(
                err instanceof Error ? err.message : "the last read failed",
            );
        });
    };
    hostTick();
    setInterval(hostTick, hostSeconds * 1000);
}
