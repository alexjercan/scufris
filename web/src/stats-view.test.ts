import { beforeEach, describe, expect, it } from "vitest";

import {
    escapeHtml,
    type Availability,
    type HostOverview,
    type HostStats,
    type UnitList,
} from "./common";
import {
    _resetStatsHistory,
    markHostCardsStale,
    renderCards,
    renderHostCards,
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
        '<div id="host-summary"></div><div id="cards"></div>' +
        '<div id="host-cards"></div>';
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

// --- host inspection cards (task 20260729-125024) ---------------------------
//
// The property under test throughout: a card NEVER renders blank. An empty but
// healthy report says "none"; an unreadable one shows its reason. Those two must
// look different, because a blank card reads as "checked, all fine" in exactly
// the case where nothing was checked.

function available(overrides: Partial<Availability> = {}): Availability {
    return { ok: true, reason: "", caveat: "", ...overrides };
}

function unitList(overrides: Partial<UnitList> = {}): UnitList {
    return {
        available: available(),
        scope: "system",
        state_filter: "failed",
        units: [],
        truncated: false,
        ...overrides,
    };
}

function fixtureOverview(overrides: Partial<HostOverview> = {}): HostOverview {
    return {
        failed_system_units: unitList(),
        failed_user_units: unitList({ scope: "user" }),
        storage: {
            available: available(),
            filesystems: {
                available: available(),
                filesystems: [
                    {
                        mountpoint: "/nix/store",
                        device: "/dev/nvme0n1p2",
                        fstype: "ext4",
                        total: 1000,
                        used: 540,
                        free: 460,
                        percent: 54,
                    },
                ],
            },
            generations: {
                available: available(),
                generations: [
                    {
                        number: 191,
                        date: "2026-07-29 16:58:12",
                        nixos_version: "26.11",
                        kernel_version: "6.18.40",
                        configuration_revision: "",
                        current: true,
                    },
                    {
                        number: 190,
                        date: "2026-07-29 16:53:30",
                        nixos_version: "26.11",
                        kernel_version: "6.18.40",
                        configuration_revision: "",
                        current: false,
                    },
                ],
            },
            nix_store: {
                mountpoint: "/nix/store",
                device: "/dev/nvme0n1p2",
                fstype: "ext4",
                total: 1000,
                used: 540,
                free: 460,
                percent: 54,
            },
        },
        thermal: {
            available: available(),
            temperatures: [
                {
                    chip: "coretemp",
                    label: "Package id 0",
                    celsius: 71,
                    high: 80,
                    critical: 100,
                },
            ],
            throttling: {
                available: available(),
                core_events: 162,
                package_events: 82,
                core_time_ms: 310,
                package_time_ms: 153,
                cpus_read: 24,
            },
            battery: { available: available(), present: false, percent: null },
            fans: { available: available(), present: false },
        },
        ...overrides,
    };
}

describe("renderHostCards", () => {
    it("renders host cards", () => {
        renderHostCards(fixtureOverview());
        const cards = [...document.querySelectorAll("#host-cards .card")];
        expect(cards).toHaveLength(4);
        const titles = cards.map((c) =>
            c.querySelector(".card__title")?.textContent?.toLowerCase(),
        );
        expect(titles.join(" ")).toContain("failed units");
        expect(titles.join(" ")).toContain("generations");
        expect(titles.join(" ")).toContain("nix store");
        expect(titles.join(" ")).toContain("thermal");
        const text = document.getElementById("host-cards")?.textContent ?? "";
        expect(text).toContain("191");
        expect(text).toContain("6.18.40");
        expect(text).toContain("71");
    });

    it("says 'none' rather than leaving the failed-units card blank", () => {
        renderHostCards(fixtureOverview());
        const card = document.querySelectorAll("#host-cards .card")[0];
        expect(card.textContent).toContain("none");
        expect(card.querySelector(".card__note")).toBeNull();
        // The count reads 0, and is not styled as a problem.
        const value = card.querySelector(".card__value");
        expect(value?.textContent).toBe("0");
        expect(value?.classList.contains("is-crit")).toBe(false);
    });

    it("names the failed units and marks the count when something failed", () => {
        renderHostCards(
            fixtureOverview({
                failed_system_units: unitList({
                    units: [
                        {
                            name: "nginx.service",
                            load: "loaded",
                            active: "failed",
                            sub: "failed",
                            description: "nginx",
                        },
                    ],
                }),
            }),
        );
        const card = document.querySelectorAll("#host-cards .card")[0];
        expect(card.textContent).toContain("nginx.service");
        expect(card.querySelector(".card__value")?.textContent).toBe("1");
        expect(card.querySelector(".card__value")?.classList).toContain(
            "is-crit",
        );
    });

    it("renders an unavailable host report with its reason", () => {
        renderHostCards(
            fixtureOverview({
                failed_system_units: unitList({
                    available: {
                        ok: false,
                        reason: "systemctl is not installed on this host",
                        caveat: "",
                    },
                }),
            }),
        );
        const card = document.querySelectorAll("#host-cards .card")[0];
        const note = card.querySelector(".card__note");
        expect(note).not.toBeNull();
        expect(note?.textContent).toContain("systemctl is not installed");
        expect(note?.classList.contains("is-error")).toBe(true);
        // Crucially it does NOT read as zero failures.
        expect(card.querySelector(".card__value")?.textContent).toBe("?");
        expect(card.textContent).not.toContain("0 failed");
    });

    it("shows a caveat on an available-but-partial report", () => {
        renderHostCards(
            fixtureOverview({
                failed_user_units: unitList({
                    scope: "user",
                    available: {
                        ok: true,
                        reason: "",
                        caveat: "2 mounts could not be read",
                    },
                }),
            }),
        );
        const card = document.querySelectorAll("#host-cards .card")[0];
        const note = card.querySelector(".card__note");
        expect(note?.textContent).toContain("2 mounts could not be read");
        // A caveat is not an error - the data IS there.
        expect(note?.classList.contains("is-error")).toBe(false);
    });

    it("distinguishes 'never throttled' from 'throttling unknown'", () => {
        const base = fixtureOverview();
        renderHostCards({
            ...base,
            thermal: {
                ...base.thermal,
                throttling: {
                    ...base.thermal.throttling,
                    core_events: 0,
                    package_events: 0,
                },
            },
        });
        let card = document.querySelectorAll("#host-cards .card")[3];
        expect(card.textContent).toContain("never since boot");

        renderHostCards({
            ...base,
            thermal: {
                ...base.thermal,
                throttling: {
                    ...base.thermal.throttling,
                    available: {
                        ok: false,
                        reason: "this CPU exposes no thermal_throttle counters",
                        caveat: "",
                    },
                },
            },
        });
        card = document.querySelectorAll("#host-cards .card")[3];
        expect(card.textContent).toContain("unknown");
        expect(card.textContent).not.toContain("never since boot");
    });

    it("reports real throttle counters, which a temperature alone cannot show", () => {
        renderHostCards(fixtureOverview());
        const card = document.querySelectorAll("#host-cards .card")[3];
        expect(card.textContent).toContain("162");
        expect(card.textContent).toContain("82");
    });

    it("renders no host cards at all when the page has no host section", () => {
        document.body.innerHTML = '<div id="cards"></div>';
        expect(() => {
            renderHostCards(fixtureOverview());
        }).not.toThrow();
    });
});

describe("renderHostCards hostile input", () => {
    // Round 1, MAJOR: the host cards fed machine-controlled strings straight
    // into innerHTML via card()/row(). A systemd unit is named by a FILE, and
    // the overview polls the user scope - so anything in ~/.config/systemd/user/
    // could name itself `<img src=x onerror=...>.service` and run script in the
    // authenticated operator's dashboard. See the ledger,
    // escape-only-host-strings-in-element-content.
    const HOSTILE = '<img src=x onerror="window.__pwned=1">';

    function expectNoMarkup(): void {
        const host = document.getElementById("host-cards");
        expect(host?.querySelectorAll("img")).toHaveLength(0);
        expect(host?.querySelectorAll("script")).toHaveLength(0);
        // ... and the value is still SHOWN, as text.
        expect(host?.textContent).toContain(HOSTILE);
    }

    it("renders a hostile unit name as text, not markup", () => {
        renderHostCards(
            fixtureOverview({
                failed_user_units: unitList({
                    scope: "user",
                    units: [
                        {
                            name: HOSTILE,
                            load: "loaded",
                            active: "failed",
                            sub: "failed",
                            description: "",
                        },
                    ],
                }),
            }),
        );
        expectNoMarkup();
    });

    it("renders a hostile mountpoint as text, not markup", () => {
        const base = fixtureOverview();
        renderHostCards({
            ...base,
            storage: {
                ...base.storage,
                nix_store: { ...base.storage.nix_store!, mountpoint: HOSTILE },
            },
        });
        expectNoMarkup();
    });

    it("renders a hostile sensor label as text, not markup", () => {
        const base = fixtureOverview();
        renderHostCards({
            ...base,
            thermal: {
                ...base.thermal,
                temperatures: [
                    {
                        chip: "coretemp",
                        label: HOSTILE,
                        celsius: 71,
                        high: 80,
                        critical: 100,
                    },
                ],
            },
        });
        expectNoMarkup();
    });

    it("renders a hostile generation date as text, not markup", () => {
        const base = fixtureOverview();
        renderHostCards({
            ...base,
            storage: {
                ...base.storage,
                generations: {
                    available: available(),
                    generations: [
                        {
                            number: 191,
                            date: HOSTILE,
                            nixos_version: "26.11",
                            kernel_version: HOSTILE,
                            configuration_revision: "",
                            current: true,
                        },
                    ],
                },
            },
        });
        expectNoMarkup();
    });

    it("renders an unavailable reason as text, not markup", () => {
        renderHostCards(
            fixtureOverview({
                failed_system_units: unitList({
                    available: { ok: false, reason: HOSTILE, caveat: "" },
                }),
            }),
        );
        expectNoMarkup();
    });
});

describe("host card truncation and staleness", () => {
    it("marks a capped failed-unit count as a lower bound", () => {
        renderHostCards(
            fixtureOverview({
                failed_system_units: unitList({
                    truncated: true,
                    units: [
                        {
                            name: "a.service",
                            load: "loaded",
                            active: "failed",
                            sub: "failed",
                            description: "",
                        },
                    ],
                }),
            }),
        );
        const card = document.querySelectorAll("#host-cards .card")[0];
        // "1" would state a capped list's length as the complete count.
        expect(card.querySelector(".card__value")?.textContent).toBe("1+");
        expect(card.textContent).toContain("lower bound");
    });

    it("marks the host section stale when a poll fails", () => {
        renderHostCards(fixtureOverview());
        markHostCardsStale("500 from /api/host/overview");
        const note = document.querySelector("#host-cards .host-stale");
        expect(note).not.toBeNull();
        expect(note?.textContent).toContain("not updating");
        expect(note?.textContent).toContain("500");
    });

    it("clears the stale marker once a poll succeeds again", () => {
        renderHostCards(fixtureOverview());
        markHostCardsStale("boom");
        renderHostCards(fixtureOverview());
        expect(document.querySelector("#host-cards .host-stale")).toBeNull();
    });

    it("does not stack a stale marker per failed poll", () => {
        renderHostCards(fixtureOverview());
        markHostCardsStale("one");
        markHostCardsStale("two");
        expect(
            document.querySelectorAll("#host-cards .host-stale"),
        ).toHaveLength(1);
    });
});
