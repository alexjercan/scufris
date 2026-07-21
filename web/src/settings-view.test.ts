import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
    AgentConfig,
    AgentConfigUpdate,
    AgentHealth,
    AgentTool,
    BackendOption,
} from "./common";
import { renderSettings } from "./settings-view";
import type { SettingsActions, SettingsExtras } from "./settings-view";

function config(over: Partial<AgentConfig> = {}): AgentConfig {
    return {
        enabled: true,
        backend: "codex",
        model: "gpt-5.5",
        auth_mode: "chatgpt",
        tools_enabled: true,
        sandbox: "read-only",
        mcp_servers: [{ id: "scufris", source: "built-in" }],
        writable: false,
        ...over,
    };
}

function tool(
    name: string,
    description = "does a thing",
    args: string[] = [],
    enabled = true,
): AgentTool {
    return { name, description, server: "scufris", args, enabled };
}

function fakeActions(over: Partial<SettingsActions> = {}): SettingsActions {
    return {
        patch: () => Promise.resolve(),
        addServer: () => Promise.resolve(),
        removeServer: () => Promise.resolve(),
        createProfile: () => Promise.resolve(),
        activateProfile: () => Promise.resolve(),
        deleteProfile: () => Promise.resolve(),
        reload: () => Promise.resolve(),
        ...over,
    };
}

function extras(over: Partial<SettingsExtras> = {}): SettingsExtras {
    return {
        sessions: { sessions: [], current: null },
        usage: null,
        context: null,
        memory: null,
        account: null,
        profiles: null,
        ...over,
    };
}

function backends(): BackendOption[] {
    return [
        { id: "codex", label: "Codex", default_model: "gpt-5.5", models: [] },
        {
            id: "claude",
            label: "Claude",
            default_model: "claude-opus-4-8",
            models: [],
        },
    ];
}

const flush = () => new Promise((r) => setTimeout(r, 0));

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
        expect(text).toContain("codex"); // backend (canonical id, not app_server)
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

    it("renders interactive controls when writable and actions are wired", () => {
        renderSettings(
            root,
            config({ writable: true }),
            [tool("host_stats")],
            null,
            fakeActions(),
            null,
            backends(),
        );
        expect(root.querySelector(".settings__toggle")).not.toBeNull();
        expect(root.querySelector(".settings__select")).not.toBeNull();
        expect(root.querySelector(".settings__input")).not.toBeNull();
        expect(root.querySelector(".settings__addserver")).not.toBeNull();
        // No stale "restart to change" copy in the writable view.
        expect(root.textContent).not.toContain("restart to change");
    });

    it("hides controls and shows a read-only banner when not writable", () => {
        renderSettings(
            root,
            config({ writable: false }),
            [],
            null,
            fakeActions(),
        );
        expect(root.textContent).toContain("Read-only server");
        expect(root.querySelector(".settings__toggle")).toBeNull();
        expect(root.querySelector(".settings__addserver")).toBeNull();
    });

    it("patches agent_enabled=false when the enabled toggle is turned off and confirmed", async () => {
        const calls: AgentConfigUpdate[] = [];
        vi.stubGlobal("confirm", () => true);
        renderSettings(
            root,
            config({ writable: true, enabled: true }),
            [],
            null,
            fakeActions({
                patch: (u) => {
                    calls.push(u);
                    return Promise.resolve();
                },
            }),
        );
        const toggle = root.querySelector(
            '.settings__toggle[aria-label="enabled"]',
        ) as HTMLInputElement;
        toggle.checked = false;
        toggle.dispatchEvent(new Event("change"));
        await flush();
        expect(calls).toEqual([{ agent_enabled: false }]);
    });

    it("does NOT patch when the disable confirm is cancelled", async () => {
        const calls: AgentConfigUpdate[] = [];
        vi.stubGlobal("confirm", () => false);
        renderSettings(
            root,
            config({ writable: true, enabled: true }),
            [],
            null,
            fakeActions({
                patch: (u) => {
                    calls.push(u);
                    return Promise.resolve();
                },
            }),
        );
        const toggle = root.querySelector(
            '.settings__toggle[aria-label="enabled"]',
        ) as HTMLInputElement;
        toggle.checked = false;
        toggle.dispatchEvent(new Event("change"));
        await flush();
        expect(calls).toEqual([]); // cancelled -> no mutation
        expect(toggle.checked).toBe(true); // reverted
    });

    it("shows friendly backend labels and patches agent_backend on change", async () => {
        const calls: AgentConfigUpdate[] = [];
        renderSettings(
            root,
            config({ writable: true, backend: "codex" }),
            [],
            null,
            fakeActions({
                patch: (u) => {
                    calls.push(u);
                    return Promise.resolve();
                },
            }),
            null,
            backends(),
        );
        const select = root.querySelector(
            '.settings__select[aria-label="backend"]',
        ) as HTMLSelectElement;
        // Options carry the friendly labels, not the raw ids.
        const labels = [...select.options].map((o) => o.textContent);
        expect(labels).toEqual(["Codex", "Claude"]);
        expect(select.value).toBe("codex"); // the current selection
        select.value = "claude";
        select.dispatchEvent(new Event("change"));
        await flush();
        expect(calls).toEqual([{ agent_backend: "claude" }]);
    });

    it("disables a tool by sending the full disabled_tools set", async () => {
        const calls: AgentConfigUpdate[] = [];
        renderSettings(
            root,
            config({ writable: true }),
            [tool("host_stats"), tool("disk_usage", "d", [], false)],
            null,
            fakeActions({
                patch: (u) => {
                    calls.push(u);
                    return Promise.resolve();
                },
            }),
        );
        // host_stats is enabled; turning it off should send both it and the
        // already-disabled disk_usage.
        const toggle = root.querySelector(
            '.tool-card__toggle[aria-label="enable host_stats"]',
        ) as HTMLInputElement;
        toggle.checked = false;
        toggle.dispatchEvent(new Event("change"));
        await flush();
        expect(calls).toHaveLength(1);
        expect(new Set(calls[0].disabled_tools)).toEqual(
            new Set(["host_stats", "disk_usage"]),
        );
    });

    it("adds an MCP server from the form and clears the inputs", async () => {
        const added: unknown[] = [];
        renderSettings(
            root,
            config({ writable: true }),
            [],
            null,
            fakeActions({
                addServer: (spec) => {
                    added.push(spec);
                    return Promise.resolve();
                },
            }),
        );
        const form = root.querySelector(
            ".settings__addserver",
        ) as HTMLFormElement;
        const [idIn, cmdIn, argsIn] = form.querySelectorAll("input");
        idIn.value = "fs";
        cmdIn.value = "mcp-fs";
        argsIn.value = "--root /tmp";
        form.dispatchEvent(new Event("submit"));
        await flush();
        expect(added).toEqual([
            { id: "fs", command: "mcp-fs", args: ["--root", "/tmp"] },
        ]);
    });

    it("removes a configured MCP server (built-in has no remove button)", async () => {
        const removed: string[] = [];
        vi.stubGlobal("confirm", () => true);
        renderSettings(
            root,
            config({
                writable: true,
                mcp_servers: [
                    { id: "scufris", source: "built-in" },
                    { id: "fs", source: "configured" },
                ],
            }),
            [],
            null,
            fakeActions({
                removeServer: (id) => {
                    removed.push(id);
                    return Promise.resolve();
                },
            }),
        );
        const buttons = root.querySelectorAll(".settings__btn--danger");
        expect(buttons).toHaveLength(1); // only the configured one
        (buttons[0] as HTMLButtonElement).dispatchEvent(new Event("click"));
        await flush();
        expect(removed).toEqual(["fs"]);
    });

    it("escapes a hostile configured server id in the writable list", () => {
        renderSettings(
            root,
            config({
                writable: true,
                mcp_servers: [
                    {
                        id: "<img src=x onerror=alert(1)>",
                        source: "configured",
                    },
                ],
            }),
            [],
            null,
            fakeActions(),
        );
        expect(root.querySelector("img")).toBeNull();
        expect(root.textContent).toContain("<img src=x onerror=alert(1)>");
    });

    it("renders the info panels with data", () => {
        renderSettings(root, config(), [], null, null, {
            sessions: {
                sessions: [
                    {
                        id: "s1",
                        title: "t",
                        started_at: null,
                        updated_at: null,
                        git_branch: null,
                        cwd: null,
                    },
                ],
                current: "s1",
            },
            usage: {
                plan_type: "plus",
                primary: {
                    used_percent: 42,
                    window_minutes: 10080,
                    resets_at: null,
                },
                secondary: null,
            },
            context: {
                session_id: "s1",
                context_window: 1000,
                input_tokens: 250,
                cached_input_tokens: 0,
                output_tokens: 0,
                reasoning_output_tokens: 0,
                total_tokens: 250,
                turn_count: 3,
                tool_call_count: 2,
            },
            memory: {
                session_count: 5,
                total_bytes: 2048,
                oldest: null,
                newest: null,
            },
            account: {
                auth_mode: "chatgpt",
                model: "gpt-5.5",
                enabled: true,
                quota: null,
            },
            profiles: null,
        });
        const text = root.textContent ?? "";
        expect(text).toContain("Sessions");
        expect(text).toContain("Usage");
        expect(text).toContain("42.0%"); // used_percent
        expect(text).toContain("25.0%"); // context window fill (250/1000)
        expect(text).toContain("Memory");
        expect(text).toContain("2.0 KB"); // memory on-disk
        expect(text).toContain("Account");
    });

    it("degrades each panel to a dash when its data is null", () => {
        renderSettings(root, config(), [], null, null, extras());
        // sessions present-but-empty -> count 0; the rest null -> dashes.
        const panels = root.querySelector(".settings__panels");
        expect(panels).not.toBeNull();
        expect(panels?.textContent).toContain("-");
        // No crash, all five panels present.
        expect(
            root.querySelectorAll(".settings__panels .settings__card").length,
        ).toBe(5);
    });

    it("omits panels entirely when no extras are passed", () => {
        renderSettings(root, config(), []);
        expect(root.querySelector(".settings__panels")).toBeNull();
    });

    it("renders a profile switcher marking the active one (writable)", () => {
        renderSettings(
            root,
            config({ writable: true }),
            [],
            null,
            fakeActions(),
            extras({
                profiles: { profiles: ["default", "cheap"], active: "default" },
            }),
        );
        expect(root.textContent).toContain("Profiles");
        const active = root.querySelector(".profiles__item--active");
        expect(active?.textContent).toContain("default");
        // The active one's activate button is disabled; the other's is enabled.
        const cheapBtn = root.querySelector(
            '.profiles__name[aria-label="activate cheap"]',
        ) as HTMLButtonElement;
        expect(cheapBtn.disabled).toBe(false);
    });

    it("activates a profile when its button is clicked", async () => {
        const activated: string[] = [];
        renderSettings(
            root,
            config({ writable: true }),
            [],
            null,
            fakeActions({
                activateProfile: (name) => {
                    activated.push(name);
                    return Promise.resolve();
                },
            }),
            extras({
                profiles: { profiles: ["default", "cheap"], active: "default" },
            }),
        );
        const btn = root.querySelector(
            '.profiles__name[aria-label="activate cheap"]',
        ) as HTMLButtonElement;
        btn.dispatchEvent(new Event("click"));
        await flush();
        expect(activated).toEqual(["cheap"]);
    });

    it("does not show the profile switcher on a read-only server", () => {
        renderSettings(
            root,
            config({ writable: false }),
            [],
            null,
            fakeActions(),
            extras({ profiles: { profiles: ["default"], active: "default" } }),
        );
        expect(root.textContent).not.toContain("Profiles");
    });

    it("creates a profile from the save-as form", async () => {
        const created: string[] = [];
        renderSettings(
            root,
            config({ writable: true }),
            [],
            null,
            fakeActions({
                createProfile: (name) => {
                    created.push(name);
                    return Promise.resolve();
                },
            }),
            extras({ profiles: { profiles: ["default"], active: "default" } }),
        );
        const input = root.querySelector(
            'input[aria-label="new profile name"]',
        ) as HTMLInputElement;
        input.value = "cheap";
        input.form?.dispatchEvent(new Event("submit"));
        await flush();
        expect(created).toEqual(["cheap"]);
    });
});
