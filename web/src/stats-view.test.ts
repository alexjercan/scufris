import { beforeEach, describe, expect, it } from "vitest";

import { escapeHtml, type HostStats } from "./common";
import { renderCards, renderSummary } from "./stats-view";

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
        ...overrides,
    };
}

beforeEach(() => {
    document.body.innerHTML =
        '<div id="host-summary"></div><div id="cards"></div>';
});

describe("escapeHtml", () => {
    it("escapes HTML metacharacters", () => {
        expect(escapeHtml('<img src=x onerror="boom">&')).toBe(
            "&lt;img src=x onerror=&quot;boom&quot;&gt;&amp;",
        );
    });
});

describe("renderCards", () => {
    it("renders one card per metric group", () => {
        renderCards(fixtureStats());
        const cards = document.querySelectorAll("#cards .card");
        // CPU, Memory, Swap, Load, Disks, Network
        expect(cards.length).toBe(6);
        expect(document.querySelector("#cards")?.textContent).toContain("40.0");
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
});

describe("renderSummary", () => {
    it("renders a hostile hostname as text, not markup", () => {
        renderSummary(fixtureStats({ hostname: "<script>alert(1)</script>" }));
        const summary = document.getElementById("host-summary");
        expect(summary?.querySelector("script")).toBeNull();
        expect(summary?.textContent).toContain("<script>alert(1)</script>");
    });
});
