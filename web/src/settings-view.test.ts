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
        parameters: [],
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
        runTool: () => Promise.resolve({ ok: true, text: "", structured: {} }),
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

describe("tool runner (try it)", () => {
    function withParams(name: string): AgentTool {
        return {
            ...tool(name),
            parameters: [
                {
                    name: "limit",
                    type: "integer",
                    required: false,
                    description: "max rows",
                    default: 15,
                },
                {
                    name: "all",
                    type: "boolean",
                    required: false,
                    description: "",
                    default: false,
                },
                {
                    name: "q",
                    type: "string",
                    required: true,
                    description: "",
                    default: null,
                },
            ],
        };
    }

    it("renders runner form from tool parameters", () => {
        root.appendChild(
            renderToolControls([withParams("list_processes")], fakeActions()),
        );
        const limit = root.querySelector(
            '.tool-runner__form input[name="limit"]',
        ) as HTMLInputElement;
        const all = root.querySelector('input[name="all"]') as HTMLInputElement;
        const q = root.querySelector('input[name="q"]') as HTMLInputElement;
        expect(limit.type).toBe("number");
        expect(all.type).toBe("checkbox");
        expect(q.type).toBe("text");
        // The required param is marked with a "*".
        const qLabel = [...root.querySelectorAll(".tool-runner__label")].find(
            (l) => l.textContent?.startsWith("q"),
        );
        expect(qLabel?.textContent).toContain("*");
        // A disabled tool gets NO runner.
        root.innerHTML = "";
        root.appendChild(
            renderToolControls([tool("disk_usage", false)], fakeActions()),
        );
        expect(root.querySelector(".tool-runner")).toBeNull();
    });

    it("run tool requires confirm", async () => {
        const calls: Array<{ name: string; args: Record<string, unknown> }> =
            [];
        const actions = fakeActions({
            runTool: (name, args) => {
                calls.push({ name, args });
                return Promise.resolve({
                    ok: true,
                    text: "done",
                    structured: {},
                });
            },
        });
        root.appendChild(
            renderToolControls([withParams("list_processes")], actions),
        );
        const form = root.querySelector(
            ".tool-runner__form",
        ) as HTMLFormElement;
        (root.querySelector('input[name="limit"]') as HTMLInputElement).value =
            "5";
        (root.querySelector('input[name="all"]') as HTMLInputElement).checked =
            true;
        (root.querySelector('input[name="q"]') as HTMLInputElement).value =
            "py";

        // Confirm denied -> the tool is NOT run.
        const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
        form.dispatchEvent(new Event("submit", { cancelable: true }));
        await flush();
        expect(calls).toHaveLength(0);

        // Confirm accepted -> runs with args coerced by declared type.
        confirmSpy.mockReturnValue(true);
        form.dispatchEvent(new Event("submit", { cancelable: true }));
        await flush();
        expect(calls).toHaveLength(1);
        expect(calls[0].name).toBe("list_processes");
        expect(calls[0].args).toEqual({ limit: 5, all: true, q: "py" });
        confirmSpy.mockRestore();
    });

    it("escapes tool run result", async () => {
        const actions = fakeActions({
            runTool: () =>
                Promise.resolve({
                    ok: true,
                    text: "<script>alert(1)</script>",
                    structured: {},
                }),
        });
        root.appendChild(renderToolControls([tool("host_stats")], actions));
        vi.spyOn(window, "confirm").mockReturnValue(true);
        const form = root.querySelector(
            ".tool-runner__form",
        ) as HTMLFormElement;
        form.dispatchEvent(new Event("submit", { cancelable: true }));
        await flush();
        const result = root.querySelector(
            ".tool-runner__result",
        ) as HTMLElement;
        // The script is inert - escaped, not a live element.
        expect(result.querySelector("script")).toBeNull();
        expect(result.innerHTML).toContain("&lt;script&gt;");
        vi.restoreAllMocks();
    });

    it("renders the error detail when a run fails", async () => {
        const actions = fakeActions({
            runTool: () => Promise.reject(new Error("tool 'x' is disabled")),
        });
        root.appendChild(renderToolControls([tool("host_stats")], actions));
        vi.spyOn(window, "confirm").mockReturnValue(true);
        const form = root.querySelector(
            ".tool-runner__form",
        ) as HTMLFormElement;
        form.dispatchEvent(new Event("submit", { cancelable: true }));
        await flush();
        const result = root.querySelector(
            ".tool-runner__result",
        ) as HTMLElement;
        expect(result.className).toContain("tool-runner__result--error");
        expect(result.textContent).toContain("disabled");
        vi.restoreAllMocks();
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
