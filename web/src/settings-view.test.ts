import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
    AgentConfig,
    AgentConfigUpdate,
    AgentHealth,
    AgentTool,
    ProfilesResponse,
} from "./common";
import {
    renderGlobalConfig,
    renderHealthCard,
    renderProfileSwitcher,
    renderServerControls,
    renderToolControls,
    type SettingsActions,
} from "./settings-view";

// These are the composable section renders reused by the unified per-agent
// settings page (agent-settings-view) for the orchestrator's GLOBAL config. The
// page composition + entry moved there; here we test each section in isolation.

function config(over: Partial<AgentConfig> = {}): AgentConfig {
    return {
        enabled: true,
        backend: "codex",
        model: "gpt-5.5",
        auth_mode: "chatgpt",
        tools_enabled: true,
        sandbox: "read-only",
        mcp_servers: [{ id: "scufris", source: "built-in" }],
        writable: true,
        ...over,
    };
}

function tool(name: string, enabled = true): AgentTool {
    return {
        name,
        description: "does a thing",
        server: "scufris",
        args: [],
        enabled,
    };
}

function health(over: Partial<AgentHealth> = {}): AgentHealth {
    return {
        scufris_version: "0.1.0",
        backend: "codex",
        backend_version: "codex 0.144",
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

const flush = () => new Promise((r) => setTimeout(r, 0));

let root: HTMLElement;
beforeEach(() => {
    document.body.innerHTML = '<main id="root"></main>';
    root = document.getElementById("root") as HTMLElement;
});

describe("renderGlobalConfig", () => {
    it("has enabled + tools toggles + read-only auth/sandbox (NOT backend/model)", () => {
        root.appendChild(renderGlobalConfig(config(), fakeActions()));
        const text = root.textContent ?? "";
        expect(text).toContain("System");
        expect(
            root.querySelector('.settings__toggle[aria-label="enabled"]'),
        ).not.toBeNull();
        expect(
            root.querySelector('.settings__toggle[aria-label="tools"]'),
        ).not.toBeNull();
        expect(text).toContain("auth mode");
        expect(text).toContain("sandbox");
        // Backend/model are the agent's record fields (in the agent-settings form),
        // not global controls - so there is NO backend select here.
        expect(
            root.querySelector('.settings__select[aria-label="backend"]'),
        ).toBeNull();
    });

    it("patches agent_tools_enabled when the tools toggle is flipped off", async () => {
        const calls: AgentConfigUpdate[] = [];
        vi.stubGlobal("confirm", () => true);
        root.appendChild(
            renderGlobalConfig(
                config(),
                fakeActions({
                    patch: (u) => {
                        calls.push(u);
                        return Promise.resolve();
                    },
                }),
            ),
        );
        const toggle = root.querySelector(
            '.settings__toggle[aria-label="tools"]',
        ) as HTMLInputElement;
        toggle.checked = false;
        toggle.dispatchEvent(new Event("change"));
        await flush();
        expect(calls).toEqual([{ agent_tools_enabled: false }]);
    });
});

describe("renderServerControls", () => {
    it("lists servers, removes a configured one, and adds a new one", async () => {
        const removed: string[] = [];
        const added: unknown[] = [];
        vi.stubGlobal("confirm", () => true);
        root.appendChild(
            renderServerControls(
                config({
                    mcp_servers: [
                        { id: "scufris", source: "built-in" },
                        { id: "fs", source: "configured" },
                    ],
                }),
                fakeActions({
                    removeServer: (id) => {
                        removed.push(id);
                        return Promise.resolve();
                    },
                    addServer: (s) => {
                        added.push(s);
                        return Promise.resolve();
                    },
                }),
            ),
        );
        // The built-in scufris has no remove button; the configured fs does.
        const rows = [...root.querySelectorAll(".settings__row")];
        const fsRow = rows.find((r) => r.textContent?.includes("fs"));
        fsRow
            ?.querySelector<HTMLButtonElement>(".settings__btn--danger")
            ?.click();
        await flush();
        expect(removed).toEqual(["fs"]);
        // Add a server from the form.
        const form = root.querySelector<HTMLFormElement>(
            ".settings__addserver",
        )!;
        (
            form.querySelector(
                '[aria-label="new MCP server id"]',
            ) as HTMLInputElement
        ).value = "extra";
        (
            form.querySelector(
                '[aria-label="new MCP server command"]',
            ) as HTMLInputElement
        ).value = "mcp-extra";
        form.dispatchEvent(new Event("submit"));
        await flush();
        expect(added).toEqual([
            { id: "extra", command: "mcp-extra", args: [] },
        ]);
    });
});

describe("renderToolControls", () => {
    it("disables a tool by sending the full disabled_tools set", async () => {
        const calls: AgentConfigUpdate[] = [];
        root.appendChild(
            renderToolControls(
                [tool("host_stats"), tool("disk_usage", false)],
                fakeActions({
                    patch: (u) => {
                        calls.push(u);
                        return Promise.resolve();
                    },
                }),
            ),
        );
        const toggle = root.querySelector(
            '.tool-card__toggle[aria-label="enable host_stats"]',
        ) as HTMLInputElement;
        toggle.checked = false;
        toggle.dispatchEvent(new Event("change"));
        await flush();
        expect(new Set(calls[0].disabled_tools)).toEqual(
            new Set(["host_stats", "disk_usage"]),
        );
    });
});

describe("renderProfileSwitcher", () => {
    it("activates a profile and creates a new one", async () => {
        const activated: string[] = [];
        const created: string[] = [];
        const profiles: ProfilesResponse = {
            profiles: ["default", "cheap"],
            active: "default",
        };
        root.appendChild(
            renderProfileSwitcher(
                profiles,
                fakeActions({
                    activateProfile: (n) => {
                        activated.push(n);
                        return Promise.resolve();
                    },
                    createProfile: (n) => {
                        created.push(n);
                        return Promise.resolve();
                    },
                }),
            ),
        );
        root.querySelector<HTMLButtonElement>(
            '.profiles__name[aria-label="activate cheap"]',
        )?.click();
        await flush();
        expect(activated).toEqual(["cheap"]);
        const form = root.querySelector<HTMLFormElement>(
            ".settings__addserver",
        )!;
        (
            form.querySelector(
                '[aria-label="new profile name"]',
            ) as HTMLInputElement
        ).value = "fast";
        form.dispatchEvent(new Event("submit"));
        await flush();
        expect(created).toEqual(["fast"]);
    });
});

describe("renderHealthCard", () => {
    it("shows the version line and each check", () => {
        root.appendChild(renderHealthCard(health()));
        const text = root.textContent ?? "";
        expect(text).toContain("Health");
        expect(text).toContain("scufris 0.1.0");
        expect(text).toContain("codex 0.144");
        expect(root.querySelector(".health__dot--ok")).not.toBeNull();
        expect(root.querySelector(".health__dot--warn")).not.toBeNull();
    });

    it("renders the backend version generically (claude, not just codex)", () => {
        root.appendChild(
            renderHealthCard(
                health({
                    backend: "claude",
                    backend_version: "claude 1.2.0",
                    checks: [
                        {
                            name: "claude cli",
                            status: "ok",
                            detail: "claude 1.2.0",
                            hint: "",
                        },
                    ],
                }),
            ),
        );
        const text = root.textContent ?? "";
        expect(text).toContain("claude 1.2.0");
        expect(text).toContain("claude cli");
    });

    it("does not inject markup from a hostile health detail", () => {
        root.appendChild(
            renderHealthCard(
                health({
                    checks: [
                        {
                            name: "x",
                            status: "error",
                            detail: "<img src=x onerror=alert(1)>",
                            hint: "",
                        },
                    ],
                }),
            ),
        );
        expect(root.querySelector("img")).toBeNull();
        expect(root.textContent).toContain("<img src=x onerror=alert(1)>");
    });
});
