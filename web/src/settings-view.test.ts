import { beforeEach, describe, expect, it } from "vitest";

import type { AgentConfig, AgentTool } from "./common";
import { renderSettings } from "./settings-view";

function config(over: Partial<AgentConfig> = {}): AgentConfig {
    return {
        enabled: true,
        backend: "app_server",
        model: "gpt-5.5",
        auth_mode: "chatgpt",
        tools_enabled: true,
        sandbox: "read-only",
        mcp_servers: [{ id: "scufris", source: "built-in" }],
        ...over,
    };
}

function tool(name: string, description = "does a thing"): AgentTool {
    return { name, description };
}

let root: HTMLElement;
beforeEach(() => {
    document.body.innerHTML = '<main id="settings"></main>';
    root = document.getElementById("settings") as HTMLElement;
});

describe("renderSettings", () => {
    it("shows the agent config, MCP servers and tool cards", () => {
        renderSettings(root, config(), [
            tool("host_stats"),
            tool("disk_usage"),
        ]);
        const text = root.textContent ?? "";
        expect(text).toContain("app_server"); // backend
        expect(text).toContain("gpt-5.5"); // model
        expect(text).toContain("read-only"); // sandbox
        expect(text).toContain("scufris"); // MCP server
        expect(text).toContain("built-in"); // server source badge

        const cards = root.querySelectorAll(".tool-card");
        expect(cards.length).toBe(2);
        expect(root.querySelector(".settings__title")?.textContent).toBe(
            "Agent",
        );
        // The tools section header counts them.
        expect(text).toContain("Tools (2)");
    });

    it("says tools are disabled instead of listing a catalog that cannot run", () => {
        // Even if the tool endpoint still enumerates them, tools_enabled=false
        // means none are callable - the page must say so, not imply a live catalog.
        renderSettings(root, config({ enabled: false, tools_enabled: false }), [
            tool("host_stats"),
            tool("disk_usage"),
        ]);
        const text = root.textContent ?? "";
        expect(text).toContain("disabled");
        expect(text).toContain("tools are disabled");
        expect(root.querySelectorAll(".tool-card").length).toBe(0);
    });

    it("shows an empty-tools message when tools are enabled but none exist", () => {
        renderSettings(root, config({ tools_enabled: true }), []);
        expect(root.textContent).toContain("no tools available.");
        expect(root.querySelectorAll(".tool-card").length).toBe(0);
    });

    it("shows a fallback when the config could not be loaded", () => {
        renderSettings(root, null, []);
        expect(root.textContent).toContain("could not load");
    });

    it("does not inject markup from a hostile tool name/description", () => {
        renderSettings(root, config(), [
            tool("<img src=x onerror=alert(1)>", "<script>alert(2)</script>"),
        ]);
        expect(root.querySelector("img")).toBeNull();
        expect(root.querySelector("script")).toBeNull();
        expect(root.textContent).toContain("<img src=x onerror=alert(1)>");
    });

    it("re-renders cleanly (no duplicate cards) on a second call", () => {
        renderSettings(root, config(), [tool("a"), tool("b")]);
        renderSettings(root, config(), [tool("c")]);
        expect(root.querySelectorAll(".tool-card").length).toBe(1);
    });
});
