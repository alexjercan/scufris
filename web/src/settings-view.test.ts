import { beforeEach, describe, expect, it } from "vitest";

import type { AgentConfig, AgentHealth, AgentTool } from "./common";
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

function tool(
    name: string,
    description = "does a thing",
    args: string[] = [],
): AgentTool {
    return { name, description, server: "scufris", args };
}

function health(over: Partial<AgentHealth> = {}): AgentHealth {
    return {
        scufris_version: "0.1.0",
        codex_version: "codex-cli 0.144.4",
        session_count: 3,
        last_session: null,
        checks: [
            { name: "agent", status: "ok", detail: "enabled", hint: "" },
            {
                name: "codex auth",
                status: "warn",
                detail: "not logged in",
                hint: "run `codex login`",
            },
        ],
        ...over,
    };
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

    it("renders the health card with status dots and versions when provided", () => {
        renderSettings(root, config(), [tool("host_stats")], health());
        const text = root.textContent ?? "";
        expect(text).toContain("Health");
        expect(text).toContain("0.1.0"); // scufris version
        expect(text).toContain("codex-cli 0.144.4"); // codex version
        expect(text).toContain("3 sessions"); // session summary
        expect(root.querySelectorAll(".health__row").length).toBe(2);
        expect(root.querySelector(".health__dot--ok")).not.toBeNull();
        expect(root.querySelector(".health__dot--warn")).not.toBeNull();
        expect(root.textContent).toContain("run `codex login`"); // fix hint
    });

    it("omits the health card when health is not available", () => {
        renderSettings(root, config(), [tool("host_stats")], null);
        expect(root.querySelector(".health__row")).toBeNull();
    });

    it("shows env-var names on config rows and server/args on tool cards", () => {
        renderSettings(root, config(), [
            tool("tatr_ls", "list tasks", ["filter", "sort"]),
        ]);
        // Env var name beside the model row.
        expect(root.textContent).toContain("SCUFRIS_AGENT_MODEL");
        // Tool card shows its server and argument names.
        const card = root.querySelector(".tool-card");
        expect(card?.querySelector(".tool-card__server")?.textContent).toBe(
            "scufris",
        );
        expect(card?.querySelector(".tool-card__args")?.textContent).toContain(
            "filter, sort",
        );
    });

    it("clamps an unknown health status to a safe dot class", () => {
        renderSettings(
            root,
            config(),
            [],
            health({
                checks: [
                    { name: "weird", status: "bogus", detail: "?", hint: "" },
                ],
            }),
        );
        // No health__dot--bogus (would be an unstyled/invisible dot); falls to warn.
        expect(root.querySelector(".health__dot--bogus")).toBeNull();
        expect(root.querySelector(".health__dot--warn")).not.toBeNull();
    });
});
