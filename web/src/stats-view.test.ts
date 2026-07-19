import { beforeEach, describe, expect, it } from "vitest";

import { escapeHtml, type HostStats } from "./common";
import {
    _resetStatsHistory,
    renderCards,
    renderSummary,
    sparkline,
} from "./stats-view";

function fixtureStats(overrides: Partial<HostStats> = {}): HostStats {
    return {
        hostname: "testbox",
        os_name: "Linux",
        kernel: "6.18.0",
        cpu_percent: 12.5,
        per_cpu_percent: [10, 15],
        mem: { total: 1000, used: 400, available: 600, percent: 40 },
        swap: { total: 200, used: 50, percent: 25 },
        disks: [{ mountpoint: "/", total: 500, used: 100, percent: 20 }],
        load_avg: [0.1, 0.2, 0.3],
        uptime_seconds: 1234,
        net: { bytes_sent: 10, bytes_recv: 20 },
        sampled_at: "2026-07-19T00:00:00Z",
        gpus: [],
        temps: [],
        fans: [],
        per_cpu_freq_mhz: [],
        net_interfaces: [],
        disk_io: [],
        process_count: 447,
        cpu_activity: { ctx_switches_per_sec: 12000, interrupts_per_sec: 9000 },
        ...overrides,
    };
}

function gpu(name: string) {
    return {
        name,
        util_percent: 4,
        mem_used_mb: 362,
        mem_total_mb: 8192,
        mem_percent: 4.4,
        temp_c: 38,
        power_w: 10,
        power_limit_w: 225,
        clock_sm_mhz: 210,
        clock_mem_mhz: 405,
    };
}

beforeEach(() => {
    document.body.innerHTML =
        '<div id="host-summary"></div><div id="cards"></div>';
    _resetStatsHistory();
});

describe("escapeHtml", () => {
    it("escapes HTML metacharacters", () => {
        expect(escapeHtml('<img src=x onerror="boom">&')).toBe(
            "&lt;img src=x onerror=&quot;boom&quot;&gt;&amp;",
        );
    });
});

describe("renderCards", () => {
    it("renders the consolidated card set", () => {
        renderCards(fixtureStats());
        const titles = [
            ...document.querySelectorAll("#cards .card__title"),
        ].map((t) => t.textContent);
        // Consolidated: CPU, Load average, Memory (incl swap), Disks, Network.
        // No standalone Swap / Temperatures / Disk IO / Network interfaces /
        // CPU frequency cards.
        expect(document.querySelectorAll("#cards .card").length).toBe(5);
        expect(titles.some((t) => t?.includes("CPU"))).toBe(true);
        expect(titles.some((t) => t?.includes("Memory"))).toBe(true);
        expect(titles.some((t) => t === "swap")).toBe(false);
        expect(titles.some((t) => t?.includes("Temperatures"))).toBe(false);
        expect(titles.some((t) => t?.includes("frequency"))).toBe(false);
    });

    it("folds swap into the Memory card", () => {
        renderCards(fixtureStats());
        const text = document.querySelector("#cards")?.textContent ?? "";
        expect(text).toContain("swap");
    });

    it("puts core temperatures on the CPU squares and folds in frequency", () => {
        renderCards(
            fixtureStats({
                per_cpu_percent: [10, 15],
                per_cpu_freq_mhz: [3000, 3200],
                temps: [
                    {
                        chip: "coretemp",
                        readings: [
                            {
                                label: "Core 0",
                                current: 67,
                                high: 90,
                                critical: 100,
                            },
                            {
                                label: "Package id 0",
                                current: 70,
                                high: 90,
                                critical: 100,
                            },
                        ],
                    },
                ],
            }),
        );
        // The core temp number is overlaid on a load square.
        const overlay = document.querySelector("#cards .core .core__temp");
        expect(overlay?.textContent).toBe("67");
        const text = document.querySelector("#cards")?.textContent ?? "";
        expect(text).toContain("GHz"); // frequency folded into the CPU card
        expect(text).toContain("package");
    });

    it("fills the Load card with tasks, activity and uptime", () => {
        renderCards(fixtureStats());
        const text = document.querySelector("#cards")?.textContent ?? "";
        expect(text).toContain("tasks");
        expect(text).toContain("447");
        expect(text).toContain("ctx switches");
        expect(text).toContain("12.0k/s");
        expect(text).toContain("uptime");
    });

    it("shows base disks statically with a dash when idle", () => {
        renderCards(
            fixtureStats({
                disk_io: [
                    { name: "nvme0n1", read_per_sec: 0, write_per_sec: 0 },
                    { name: "nvme0n1p1", read_per_sec: 0, write_per_sec: 0 },
                    { name: "loop0", read_per_sec: 5, write_per_sec: 5 },
                ],
            }),
        );
        const text = document.querySelector("#cards")?.textContent ?? "";
        // Base disk shown even when idle...
        expect(text).toContain("nvme0n1");
        expect(text).toContain("-");
        // ...partitions and loop noise dropped.
        expect(text).not.toContain("nvme0n1p1");
        expect(text).not.toContain("loop0");
    });

    it("puts disk IO and disk temperature in the Disks card", () => {
        renderCards(
            fixtureStats({
                disk_io: [
                    {
                        name: "nvme0n1",
                        read_per_sec: 1000,
                        write_per_sec: 2000,
                    },
                ],
                temps: [
                    {
                        chip: "nvme",
                        readings: [
                            {
                                label: "Composite",
                                current: 41,
                                high: null,
                                critical: null,
                            },
                        ],
                    },
                ],
            }),
        );
        const text = document.querySelector("#cards")?.textContent ?? "";
        expect(text).toContain("nvme0n1"); // IO
        expect(text).toContain("nvme Composite"); // temp routed into Disks
    });

    it("renders one card per GPU", () => {
        renderCards(fixtureStats({ gpus: [gpu("NVIDIA RTX 3060 Ti")] }));
        const text = document.querySelector("#cards")?.textContent ?? "";
        expect(text).toContain("NVIDIA RTX 3060 Ti");
    });

    it("does not inject markup from a hostile disk mountpoint", () => {
        renderCards(
            fixtureStats({
                disks: [
                    {
                        mountpoint: "/<img src=x onerror=alert(1)>",
                        total: 500,
                        used: 100,
                        percent: 20,
                    },
                ],
            }),
        );
        expect(document.querySelector("#cards img")).toBeNull();
        expect(document.querySelector("#cards")?.textContent).toContain(
            "/<img src=x onerror=alert(1)>",
        );
    });

    it("does not inject markup from a hostile GPU name", () => {
        renderCards(
            fixtureStats({ gpus: [gpu("<img src=x onerror=alert(1)>")] }),
        );
        expect(document.querySelector("#cards img")).toBeNull();
        expect(document.querySelector("#cards")?.textContent).toContain(
            "<img src=x onerror=alert(1)>",
        );
    });
});

describe("sparkline", () => {
    it("draws one point per value with an area and a line", () => {
        const svg = sparkline([10, 20, 30], 100);
        const line = svg.querySelector(".spark__line");
        const area = svg.querySelector(".spark__area");
        expect(line).not.toBeNull();
        expect(area).not.toBeNull();
        const pts = line?.getAttribute("points")?.split(" ") ?? [];
        expect(pts.length).toBe(3);
    });

    it("is empty-safe (no polyline) for no data", () => {
        const svg = sparkline([], 100);
        expect(svg.querySelector(".spark__line")).toBeNull();
        expect(svg.querySelector(".spark__area")).toBeNull();
    });

    it("clamps a value above max to the top of the graph (y=0)", () => {
        const svg = sparkline([200], 100);
        // Single point sits at the right edge; over-max clamps to the top.
        expect(svg.querySelector(".spark__line")?.getAttribute("points")).toBe(
            "100.0,0.0",
        );
    });

    it("carries the severity class on the svg root", () => {
        const svg = sparkline([95], 100, "is-crit");
        expect(svg.getAttribute("class")).toContain("is-crit");
    });

    it("adds a <title> tooltip when given, even with no data", () => {
        const svg = sparkline([], 100, "", "CPU utilization (%)");
        expect(svg.querySelector("title")?.textContent).toBe(
            "CPU utilization (%)",
        );
    });
});

describe("sparkline history", () => {
    it("grows the CPU graph by one point per poll", () => {
        renderCards(fixtureStats({ cpu_percent: 10 }));
        renderCards(fixtureStats({ cpu_percent: 20 }));
        // CPU is the first card; its sparkline should now hold two samples.
        const line = document.querySelector("#cards .card .spark__line");
        const pts = line?.getAttribute("points")?.split(" ") ?? [];
        expect(pts.length).toBe(2);
    });

    it("gives every card a mini-graph", () => {
        renderCards(fixtureStats({ gpus: [gpu("NVIDIA RTX 3060 Ti")] }));
        // CPU, Load, GPU, Memory, Disks, Network - one .spark each.
        expect(document.querySelectorAll("#cards .card .spark").length).toBe(6);
    });

    it("labels each graph with a corner caption and a hover tooltip", () => {
        renderCards(fixtureStats());
        // CPU is the first card.
        const cpuCard = document.querySelector("#cards .card");
        expect(cpuCard?.querySelector(".spark__label")?.textContent).toBe(
            "cpu %",
        );
        expect(cpuCard?.querySelector(".spark title")?.textContent).toBe(
            "CPU utilization (%)",
        );
    });
});

describe("GPU card layout", () => {
    it("puts the VRAM fill bar directly below the vram numbers", () => {
        renderCards(fixtureStats({ gpus: [gpu("NVIDIA RTX 3060 Ti")] }));
        const rows = [...document.querySelectorAll("#cards .card .row")];
        const vramRow = rows.find((r) => r.textContent?.includes("vram"));
        expect(vramRow).toBeDefined();
        // The bar is the vram row's immediate next sibling (below, not above).
        expect(vramRow?.nextElementSibling?.classList.contains("bar")).toBe(
            true,
        );
    });
});

describe("renderSummary", () => {
    it("renders a hostile hostname as text, not markup", () => {
        renderSummary(fixtureStats({ hostname: "<script>alert(1)</script>" }));
        const summary = document.getElementById("host-summary");
        expect(summary?.querySelector("script")).toBeNull();
        expect(summary?.textContent).toContain("<script>alert(1)</script>");
    });
});
