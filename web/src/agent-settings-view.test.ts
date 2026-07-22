import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
    AccountInfo,
    Agent,
    AgentHealth,
    AgentRunStatus,
    BackendOption,
    MemoryFootprint,
    UsageQuota,
} from "./common";
import {
    agentSettingsDeps,
    createAgentSettings,
    renderAgentSettings,
    type AgentSettingsData,
    type AgentSettingsDeps,
    type AgentSettingsGlobal,
} from "./agent-settings-view";

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

function backends(): BackendOption[] {
    return [
        {
            id: "codex",
            label: "Codex",
            default_model: "gpt-5.5",
            models: ["gpt-5.5", "gpt-5.6"],
        },
        {
            id: "claude",
            label: "Claude",
            default_model: "claude-opus-4-8",
            models: ["claude-opus-4-8"],
        },
    ];
}

function health(): AgentHealth {
    return {
        scufris_version: "0.1.0",
        backend: "codex",
        backend_version: "codex 0.144",
        session_count: 2,
        last_session: null,
        checks: [{ name: "agent", status: "ok", detail: "enabled", hint: "" }],
    };
}

function status(over: Partial<AgentRunStatus> = {}): AgentRunStatus {
    return {
        agent_id: "builder",
        state: "running",
        session_id: "s1",
        turns: 3,
        tool_calls: 2,
        input_tokens: 12000,
        output_tokens: 40,
        context_window: 200000,
        last_message: null,
        updated_at: null,
        ...over,
    };
}

function usage(): UsageQuota {
    return {
        plan_type: "plus",
        primary: { used_percent: 34, window_minutes: 10080, resets_at: null },
        secondary: null,
    };
}

function memory(): MemoryFootprint {
    return {
        session_count: 5,
        total_bytes: 2048,
        oldest: null,
        newest: null,
    };
}

function account(over: Partial<AccountInfo> = {}): AccountInfo {
    return {
        auth_mode: "chatgpt",
        model: "gpt-5.5",
        enabled: true,
        quota: usage(),
        ...over,
    };
}

function data(over: Partial<AgentSettingsData> = {}): AgentSettingsData {
    return {
        agent: agent(),
        project: {
            id: "my-app",
            cwd: "/x",
            name: "My App",
            language: "python",
            description: "",
        },
        backends: backends(),
        health: health(),
        status: status(),
        usage: usage(),
        memory: memory(),
        account: account(),
        sessions: null,
        global: null,
        writable: true,
        ...over,
    };
}

// The orchestrator's global config sections (system toggles + MCP + tools +
// profiles). A project agent's `data.global` stays null.
function globalSections(
    over: Partial<AgentSettingsGlobal> = {},
): AgentSettingsGlobal {
    return {
        config: {
            enabled: true,
            backend: "codex",
            model: "gpt-5.5",
            auth_mode: "chatgpt",
            tools_enabled: true,
            sandbox: "read-only",
            mcp_servers: [{ id: "scufris", source: "built-in" }],
            writable: true,
        },
        tools: [
            {
                name: "host_stats",
                description: "host",
                server: "scufris",
                args: [],
                enabled: true,
            },
        ],
        profiles: { profiles: ["default"], active: "default" },
        actions: {
            patch: () => Promise.resolve(),
            addServer: () => Promise.resolve(),
            removeServer: () => Promise.resolve(),
            createProfile: () => Promise.resolve(),
            activateProfile: () => Promise.resolve(),
            deleteProfile: () => Promise.resolve(),
            reload: () => Promise.resolve(),
        },
        ...over,
    };
}

function deps(over: Partial<AgentSettingsDeps> = {}): AgentSettingsDeps {
    return {
        load: () => Promise.resolve(data()),
        save: () => Promise.resolve(),
        ...over,
    };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

let root: HTMLElement;
beforeEach(() => {
    document.body.innerHTML = '<main id="root"></main>';
    root = document.getElementById("root") as HTMLElement;
});

describe("renderAgentSettings", () => {
    it("renders the editable fields form (prefilled) + health + panels", () => {
        renderAgentSettings(root, data(), deps());
        const text = root.textContent ?? "";
        expect(text).toContain("Builder"); // agent name heading
        // Editable fields prefilled.
        const name = root.querySelector<HTMLInputElement>(
            'input[aria-label="agent settings name"]',
        );
        const model = root.querySelector<HTMLInputElement>(
            'input[aria-label="agent settings model"]',
        );
        expect(name?.value).toBe("Builder");
        expect(model?.value).toBe("gpt-5.5");
        // Health + the detailed panels are all present.
        expect(text).toContain("Health");
        expect(text).toContain("this session"); // status/context panel
        expect(text).toContain("account"); // account panel
        expect(text).toContain("account usage"); // usage panel
        expect(text).toContain("on-disk memory"); // memory panel
        expect(text).toContain("34%"); // usage percent
        expect(text).toContain("5"); // memory sessions
        // A back-to-chat link.
        expect(
            root
                .querySelector<HTMLAnchorElement>(".agents__back")
                ?.getAttribute("href"),
        ).toBe("/agents/builder");
    });

    it("saves the edited fields on submit", async () => {
        const save = vi.fn(() => Promise.resolve());
        renderAgentSettings(root, data(), deps({ save }));
        const name = root.querySelector<HTMLInputElement>(
            'input[aria-label="agent settings name"]',
        );
        name!.value = "Renamed";
        root.querySelector<HTMLFormElement>(
            ".settings__addserver",
        )?.dispatchEvent(new Event("submit"));
        await flush();
        expect(save).toHaveBeenCalledWith(
            expect.objectContaining({ name: "Renamed", backend: "codex" }),
        );
    });

    it("does not save when the name is blanked", async () => {
        const save = vi.fn(() => Promise.resolve());
        renderAgentSettings(root, data(), deps({ save }));
        const name = root.querySelector<HTMLInputElement>(
            'input[aria-label="agent settings name"]',
        );
        name!.value = "   ";
        root.querySelector<HTMLFormElement>(
            ".settings__addserver",
        )?.dispatchEvent(new Event("submit"));
        await flush();
        expect(save).not.toHaveBeenCalled();
    });

    it("renders a read-only view when not writable (no form)", () => {
        renderAgentSettings(root, data({ writable: false }), deps());
        expect(root.querySelector("form")).toBeNull();
        const text = root.textContent ?? "";
        expect(text).toContain("Read-only server");
        expect(text).toContain("permission mode");
        expect(text).toContain("manual");
    });

    it("renders for the orchestrator (projectless) and codex-null panels", () => {
        renderAgentSettings(
            root,
            data({
                agent: agent({
                    id: "orchestrator",
                    name: "Orchestrator",
                    project_id: "",
                }),
                project: null,
                usage: null, // e.g. a claude/mock agent has no codex account data
                account: account({ quota: null }),
            }),
            deps(),
        );
        const text = root.textContent ?? "";
        expect(text).toContain("Orchestrator");
        expect(text).toContain("server dir");
        // A null panel shows a dash, not a crash.
        expect(text).toContain("-");
    });

    it("shows the GLOBAL config sections only when data.global is set (orchestrator)", () => {
        // A project agent (global null) has NO System/MCP/Profiles sections.
        renderAgentSettings(root, data(), deps());
        let text = root.textContent ?? "";
        expect(text).not.toContain("System");
        expect(text).not.toContain("MCP servers");
        expect(text).not.toContain("Profiles");
        // The orchestrator (global present) shows them.
        renderAgentSettings(
            root,
            data({
                agent: agent({ id: "orchestrator", name: "Orchestrator" }),
                global: globalSections(),
            }),
            deps(),
        );
        text = root.textContent ?? "";
        expect(text).toContain("System"); // enabled + tools toggles
        expect(text).toContain("MCP servers");
        expect(text).toContain("scufris"); // the built-in MCP server row
        expect(text).toContain("Profiles");
        expect(text).toContain("host_stats"); // the tools catalog
    });

    it("hides the global sections on a read-only server even for the orchestrator", () => {
        renderAgentSettings(
            root,
            data({
                agent: agent({ id: "orchestrator", name: "Orchestrator" }),
                global: globalSections(),
                writable: false,
            }),
            deps(),
        );
        expect(root.textContent).not.toContain("MCP servers");
        expect(root.textContent).toContain("Read-only server");
    });

    it("shows the Sessions section only when data.sessions is set (orchestrator)", () => {
        // A project agent (sessions null) has NO Sessions section. NOTE: this
        // negative is case-sensitive - the memory panel has a lowercase
        // "sessions" row, so only the capitalized panel title is asserted absent.
        renderAgentSettings(root, data(), deps());
        expect(root.textContent ?? "").not.toContain("Sessions");
        // The orchestrator (sessions present) shows the count, current title,
        // and a link to the landing chat where the switcher lives.
        renderAgentSettings(
            root,
            data({
                agent: agent({ id: "orchestrator", name: "Orchestrator" }),
                sessions: {
                    sessions: [
                        {
                            id: "s1",
                            title: "first chat",
                            started_at: null,
                            updated_at: null,
                            git_branch: null,
                            cwd: null,
                        },
                        {
                            id: "s2",
                            title: "second chat",
                            started_at: null,
                            updated_at: null,
                            git_branch: null,
                            cwd: null,
                        },
                    ],
                    current: "s2",
                },
            }),
            deps(),
        );
        const text = root.textContent ?? "";
        expect(text).toContain("Sessions");
        // The count, asserted in its own row cell (not a bare "2" that could
        // match a timestamp or percent elsewhere on the page).
        const countRow = [...root.querySelectorAll(".settings__row")].find(
            (r) => r.querySelector(".settings__key")?.textContent === "count",
        );
        expect(countRow?.querySelector(".settings__val")?.textContent).toBe(
            "2",
        );
        expect(text).toContain("second chat"); // the current session's title
        const link = [...root.querySelectorAll("a")].find(
            (a) => a.getAttribute("href") === "/",
        );
        expect(link).toBeTruthy();
    });

    it("points the orchestrator's back-to-chat link at / (not /agents/...)", () => {
        renderAgentSettings(
            root,
            data({
                agent: agent({ id: "orchestrator", name: "Orchestrator" }),
            }),
            deps(),
        );
        expect(
            root
                .querySelector<HTMLAnchorElement>(".agents__back")
                ?.getAttribute("href"),
        ).toBe("/");
    });

    it("shows a fallback for an unknown agent", () => {
        renderAgentSettings(root, data({ agent: null }), deps());
        expect(root.textContent).toContain("no such agent.");
    });

    it("escapes a hostile agent name + description", () => {
        renderAgentSettings(
            root,
            data({
                agent: agent({
                    name: "<img src=x onerror=alert(1)>",
                    description: "<script>alert(2)</script>",
                }),
            }),
            deps(),
        );
        expect(root.querySelector("img")).toBeNull();
        expect(root.querySelector("script")).toBeNull();
        expect(root.textContent).toContain("<img src=x onerror=alert(1)>");
    });
});

describe("agentSettingsDeps", () => {
    it("loads health from the PER-AGENT url (not the global /api/agent/health)", async () => {
        const urls: string[] = [];
        const fetchMock = vi.fn((input: RequestInfo | URL) => {
            const url =
                typeof input === "string"
                    ? input
                    : input instanceof URL
                      ? input.href
                      : input.url;
            urls.push(url);
            // Every GET returns an empty-ish 200 so load() resolves; the agent
            // fetch needs a real object so the panels build.
            const body = url.endsWith("/api/agents/builder")
                ? { id: "builder", project_id: "" }
                : {};
            return Promise.resolve(
                new Response(JSON.stringify(body), {
                    status: 200,
                    headers: { "Content-Type": "application/json" },
                }),
            );
        });
        vi.stubGlobal("fetch", fetchMock);
        try {
            await agentSettingsDeps("builder").load(() => {});
        } finally {
            vi.unstubAllGlobals();
        }
        expect(urls).toContain("/api/agents/builder/health");
        expect(urls).not.toContain("/api/agent/health");
    });
});

describe("createAgentSettings", () => {
    it("loads then renders, and re-loads after a save", async () => {
        const load = vi.fn(() => Promise.resolve(data()));
        const save = vi.fn(() => Promise.resolve());
        createAgentSettings(root, { load, save });
        await flush();
        expect(load).toHaveBeenCalledTimes(1);
        expect(root.textContent).toContain("Builder");
        // A save triggers a re-load (fresh data after the write).
        root.querySelector<HTMLInputElement>(
            'input[aria-label="agent settings name"]',
        )!.value = "New";
        root.querySelector<HTMLFormElement>(
            ".settings__addserver",
        )?.dispatchEvent(new Event("submit"));
        await flush();
        expect(save).toHaveBeenCalledOnce();
        expect(load).toHaveBeenCalledTimes(2);
    });
});
