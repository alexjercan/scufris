import { beforeEach, describe, expect, it } from "vitest";

import {
    renderProcesses,
    _resetProcessState,
    type ProcessGroup,
    type ProcessList,
} from "./processes-view";

function group(
    name: string,
    cpu: number,
    mem: number,
    instances: ProcessGroup["instances"] = [],
): ProcessGroup {
    return {
        name,
        count: instances.length || 1,
        cpu_percent: cpu,
        mem_rss: mem,
        instances,
    };
}

function fixtureList(): ProcessList {
    return {
        total: 3,
        groups: [
            group("rustc", 50, 50, [
                {
                    pid: 3,
                    username: "alex",
                    cpu_percent: 50,
                    mem_rss: 50,
                    num_threads: 4,
                    status: "running",
                },
            ]),
            group("firefox", 30, 300, [
                {
                    pid: 2,
                    username: "alex",
                    cpu_percent: 20,
                    mem_rss: 200,
                    num_threads: 8,
                    status: "running",
                },
                {
                    pid: 1,
                    username: "alex",
                    cpu_percent: 10,
                    mem_rss: 100,
                    num_threads: 8,
                    status: "sleeping",
                },
            ]),
        ],
    };
}

beforeEach(() => {
    document.body.innerHTML = '<section id="processes"></section>';
    _resetProcessState();
});

describe("renderProcesses", () => {
    it("renders one row per group, sorted by cpu by default", () => {
        renderProcesses(fixtureList());
        const names = [
            ...document.querySelectorAll(".proc__group .proc__name"),
        ].map((n) => n.textContent);
        expect(names).toEqual(["rustc", "firefox"]);
        expect(document.querySelector("#processes")?.textContent).toContain(
            "3 total",
        );
    });

    it("expands a group to reveal its instances on click", () => {
        renderProcesses(fixtureList());
        expect(document.querySelectorAll(".proc__inst").length).toBe(0);
        // firefox is the second group by default cpu sort.
        const firefox =
            document.querySelectorAll<HTMLElement>(".proc__group")[1];
        firefox.click();
        const instances = document.querySelectorAll(".proc__inst");
        expect(instances.length).toBe(2);
        expect(document.querySelector("#processes")?.textContent).toContain(
            "2 alex",
        );
    });

    it("re-sorts by memory when MEM is clicked", () => {
        renderProcesses(fixtureList());
        const memBtn = [
            ...document.querySelectorAll<HTMLElement>(".proc__sortbtn"),
        ].find((b) => b.textContent === "MEM");
        memBtn?.click();
        const names = [
            ...document.querySelectorAll(".proc__group .proc__name"),
        ].map((n) => n.textContent);
        // firefox (300) now before rustc (50).
        expect(names).toEqual(["firefox", "rustc"]);
    });

    it("does not inject markup from a hostile process name", () => {
        const list = fixtureList();
        list.groups[0].name = "<img src=x onerror=alert(1)>";
        renderProcesses(list);
        expect(document.querySelector("#processes img")).toBeNull();
        expect(document.querySelector("#processes")?.textContent).toContain(
            "<img src=x onerror=alert(1)>",
        );
    });
});
