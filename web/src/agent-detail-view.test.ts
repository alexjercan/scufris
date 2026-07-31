import { beforeEach, describe, expect, it } from "vitest";

import type { Agent, AgentRunStatus, Project } from "./agent-types";
import { agentIdFromPath, renderSidebar } from "./agent-detail-view";

function agent(over: Partial<Agent> = {}): Agent {
    return {
        id: "builder",
        name: "Builder",
        project_id: "my-app",
        backend: "codex",
        model: "gpt-5.5",
        description: "does helpful things",
        goal: "",
        task_id: "",
        session_id: null,
        state: "idle",
        permission_mode: "manual",
        ...over,
    };
}

function project(): Project {
    return {
        id: "my-app",
        cwd: "/home/alex/personal/my-app",
        name: "My App",
        language: "python",
        description: "a thing",
    };
}

function status(over: Partial<AgentRunStatus> = {}): AgentRunStatus {
    return {
        agent_id: "builder",
        state: "running",
        session_id: "sess-1",
        turns: 3,
        tool_calls: 2,
        input_tokens: 1000,
        output_tokens: 40,
        context_window: 200000,
        last_message: "working on it",
        updated_at: 1785074524,
        ...over,
    };
}

let root: HTMLElement;
beforeEach(() => {
    document.body.innerHTML = '<div id="root"></div>';
    root = document.getElementById("root") as HTMLElement;
});

describe("agentIdFromPath", () => {
    it("parses /agents/<id> and /agents/<id>/settings", () => {
        expect(agentIdFromPath("/agents/builder")).toBe("builder");
        expect(agentIdFromPath("/agents/builder/settings")).toBe("builder");
        expect(agentIdFromPath("/agents/my%20agent")).toBe("my agent");
    });
    it("returns null for the list or non-agent paths", () => {
        expect(agentIdFromPath("/agents/")).toBeNull();
        expect(agentIdFromPath("/projects/x")).toBeNull();
    });
});

describe("renderSidebar", () => {
    it("renders header + a Settings LINK + status/context boxes, no form", () => {
        renderSidebar(root, agent(), project(), status());
        const text = root.textContent ?? "";
        expect(text).toContain("Builder"); // agent name
        expect(text).toContain("My App"); // project
        expect(root.querySelector(".agents__back")).not.toBeNull();
        expect(root.querySelector(".agents__badge")).not.toBeNull();
        // Chat-first: no settings form here (it lives on the settings PAGE).
        expect(root.querySelector("form")).toBeNull();
        const heads = [...root.querySelectorAll(".usage-block__head")].map(
            (h) => h.textContent,
        );
        expect(heads).toContain("status");
        expect(heads).toContain("context");
        // The Settings affordance is a LINK to the per-agent settings page.
        const link = root.querySelector<HTMLAnchorElement>(
            'a[aria-label="open settings"]',
        );
        expect(link?.getAttribute("href")).toBe("/agents/builder/settings");
        expect(text.toLowerCase()).not.toContain("session");
    });

    it("shows the running turns/tools in the status box", () => {
        renderSidebar(root, agent(), project(), status());
        const rows = [...root.querySelectorAll(".usage-block .row")];
        const byKey = (k: string) =>
            rows.find((r) => r.querySelector("span")?.textContent === k);
        expect(
            byKey("turns")?.querySelector("span:last-child")?.textContent,
        ).toBe("3");
        expect(
            byKey("tools")?.querySelector("span:last-child")?.textContent,
        ).toBe("2");
    });

    it("shows 'not started' for a never-run agent", () => {
        renderSidebar(
            root,
            agent(),
            project(),
            status({ state: "idle", session_id: null, turns: 0 }),
        );
        expect(root.textContent).toContain("not started");
    });

    it("shows a fallback for an unknown agent", () => {
        renderSidebar(root, null, null, null);
        expect(root.textContent).toContain("no such agent.");
    });

    it("gives the orchestrator a Settings link too (now editable) + 'server dir'", () => {
        renderSidebar(
            root,
            agent({ id: "orchestrator", name: "Orchestrator", project_id: "" }),
            null,
            status(),
        );
        const link = root.querySelector<HTMLAnchorElement>(
            'a[aria-label="open settings"]',
        );
        expect(link?.getAttribute("href")).toBe(
            "/agents/orchestrator/settings",
        );
        expect(root.textContent).toContain("server dir");
    });

    it("escapes a hostile agent name", () => {
        renderSidebar(
            root,
            agent({ name: '<img src=x onerror="alert(1)">' }),
            project(),
            status(),
        );
        expect(root.querySelector("img")).toBeNull();
        expect(root.textContent).toContain("<img");
    });
});
