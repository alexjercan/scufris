import { beforeEach, describe, expect, it } from "vitest";

import type { Agent, AgentRunStatus, Project } from "./common";
import { agentIdFromPath, renderAgentDetail } from "./agent-detail-view";

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
        turns: 2,
        tool_calls: 1,
        input_tokens: 100,
        output_tokens: 20,
        context_window: 258400,
        last_message: "working on it",
        updated_at: 1785074524,
        ...over,
    };
}

let root: HTMLElement;
beforeEach(() => {
    document.body.innerHTML = '<main id="agent-detail"></main>';
    root = document.getElementById("agent-detail") as HTMLElement;
});

describe("agentIdFromPath", () => {
    it("parses /agents/<id> and /agents/<id>/settings", () => {
        expect(agentIdFromPath("/agents/builder")).toBe("builder");
        expect(agentIdFromPath("/agents/builder/")).toBe("builder");
        expect(agentIdFromPath("/agents/builder/settings")).toBe("builder");
        expect(agentIdFromPath("/agents/my%20agent")).toBe("my agent");
    });

    it("returns null for the list or non-agent paths", () => {
        expect(agentIdFromPath("/agents/")).toBeNull();
        expect(agentIdFromPath("/agents")).toBeNull();
        expect(agentIdFromPath("/projects/x")).toBeNull();
    });
});

describe("renderAgentDetail", () => {
    it("renders the agent's read-only fields + a back link", () => {
        renderAgentDetail(root, agent(), project(), status());
        const text = root.textContent ?? "";
        expect(text).toContain("Builder");
        expect(text).toContain("My App"); // resolved project name
        expect(text).toContain("does helpful things"); // description
        expect(text).toContain("manual"); // permission mode
        expect(text).toContain("working on it"); // last message
        const back = root.querySelector<HTMLAnchorElement>(".agents__back");
        expect(back?.getAttribute("href")).toBe("/agents/");
    });

    it("shows a fallback for an unknown agent", () => {
        renderAgentDetail(root, null, null, null);
        expect(root.textContent).toContain("no such agent.");
    });

    it("shows 'not started' for a never-run agent", () => {
        renderAgentDetail(
            root,
            agent(),
            project(),
            status({ state: "idle", session_id: null, turns: 0 }),
        );
        expect(root.textContent).toContain("not started");
    });

    it("escapes a hostile agent name/description", () => {
        renderAgentDetail(
            root,
            agent({
                name: '<img src=x onerror="alert(1)">',
                description: "<script>alert(2)</script>",
            }),
            project(),
            status(),
        );
        expect(root.querySelector("img")).toBeNull();
        expect(root.querySelector("script")).toBeNull();
        expect(root.innerHTML).toContain("&lt;script&gt;");
    });
});
