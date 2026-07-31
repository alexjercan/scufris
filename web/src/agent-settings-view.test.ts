import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
    AccountInfo,
    Agent,
    AgentHealth,
    AgentRunStatus,
    AgentTool,
    BackendOption,
    MemoryFootprint,
    UsageQuota,
} from "./agent-types";
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
        mcpServers: [],
        capabilities: { skills: [], tools: [] },
        writable: true,
        ...over,
    };
}

// The orchestrator's global actions feed writable tool toggles. A project
// agent's `data.global` stays null.
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
            writable: true,
        },
        actions: {
            patch: () => Promise.resolve(),
            runTool: () =>
                Promise.resolve({ ok: true, text: "", structured: {} }),
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

    it("renders the account auth mode as a human label, per backend", () => {
        // A claude agent shows claude.ai (not the raw wire value, not codex's).
        renderAgentSettings(
            root,
            data({ account: account({ auth_mode: "claude_ai" }) }),
            deps(),
        );
        expect(root.textContent).toContain("claude.ai");
        expect(root.textContent).not.toContain("claude_ai");
        // A codex agent shows ChatGPT.
        renderAgentSettings(
            root,
            data({ account: account({ auth_mode: "chatgpt" }) }),
            deps(),
        );
        expect(root.textContent).toContain("ChatGPT");
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

    it("renders a project agent's role-scoped tools panel (read-only)", () => {
        const requestInput: AgentTool = {
            name: "request_input",
            description: "signal the orchestrator you are blocked",
            server: "scufris",
            args: ["question"],
            parameters: [],
            enabled: true,
        };
        renderAgentSettings(
            root,
            data({
                mcpServers: [
                    {
                        id: "agent",
                        status: "ok",
                        detail: "1 tool",
                        tools: [requestInput],
                    },
                ],
            }),
            deps(),
        );
        const text = root.textContent ?? "";
        // The grouped "MCP tools" section: one server block per server, each a
        // tool-card grid. The card carries the cyan name, the server badge, the
        // args line, and a per-tool bulb.
        expect(text).toContain("MCP tools");
        const block = root.querySelector(".mcp-server");
        expect(block).not.toBeNull();
        expect(root.querySelector(".mcp-server__id")?.textContent).toBe(
            "agent",
        );
        const cards = block?.querySelectorAll(".tool-card") ?? [];
        expect(cards.length).toBe(1);
        expect(root.querySelector(".tool-card__name")?.textContent).toBe(
            "request_input",
        );
        expect(root.querySelector(".tool-card__server")?.textContent).toBe(
            "scufris",
        );
        expect(root.querySelector(".tool-card__args")?.textContent).toContain(
            "question",
        );
        // Health circles live in the Health card now - no bulb/status dot here.
        expect(root.querySelector(".tool-card__bulb")).toBeNull();
        expect(root.querySelector(".mcp-server .health__dot")).toBeNull();
        // Read-only: no toggle/checkbox controls and no "try it" runner (those are
        // the orchestrator's writable operator console).
        expect(root.querySelector('input[type="checkbox"]')).toBeNull();
        expect(root.querySelector(".tool-runner")).toBeNull();
    });

    it("shows a 'none' tools note when the agent's backend has no scufris tools", () => {
        renderAgentSettings(root, data({ mcpServers: [] }), deps());
        expect(root.textContent ?? "").toContain(
            "none (this backend exposes no",
        );
    });

    it("does not render the read-only per-agent tools panel for the orchestrator", () => {
        // The orchestrator uses the writable operator console (global), not the
        // read-only per-agent panel, so it never shows the 'none' note even with
        // an empty per-agent tool list.
        renderAgentSettings(
            root,
            data({
                agent: agent({
                    id: "orchestrator",
                    name: "Orchestrator",
                    project_id: "",
                }),
                mcpServers: [],
            }),
            deps(),
        );
        expect(root.textContent ?? "").not.toContain(
            "none (this backend exposes no",
        );
    });

    it("renders project skills + tools cards for a project agent", () => {
        renderAgentSettings(
            root,
            data({
                capabilities: {
                    skills: [
                        {
                            name: "deploy",
                            description: "Ship the app",
                            source: ".claude/skills/deploy/SKILL.md",
                        },
                    ],
                    tools: [
                        {
                            name: "fs",
                            description: "npx fs-server",
                            source: ".mcp.json",
                            kind: "stdio",
                        },
                    ],
                },
            }),
            deps(),
        );
        const text = root.textContent ?? "";
        expect(text).toContain("Project skills (1)");
        expect(text).toContain("deploy");
        expect(text).toContain("Ship the app");
        expect(text).toContain("Project tools (1)");
        expect(text).toContain("fs");
        expect(text).toContain("npx fs-server");
        expect(text).toContain("stdio");
        // Read-only: no inputs/controls in the project cards.
        expect(root.querySelector('input[type="checkbox"]')).toBeNull();
    });

    it("shows explicit empty-state cards when a project defines no skills/tools", () => {
        renderAgentSettings(
            root,
            data({ capabilities: { skills: [], tools: [] } }),
            deps(),
        );
        const text = root.textContent ?? "";
        expect(text).toContain("none (this project defines no skills)");
        expect(text).toContain("none (this project defines no tools)");
    });

    it("renders NEITHER project card when the agent has no project (null capabilities)", () => {
        renderAgentSettings(
            root,
            data({
                agent: agent({
                    id: "orchestrator",
                    name: "Orchestrator",
                    project_id: "",
                }),
                project: null,
                capabilities: null,
            }),
            deps(),
        );
        const text = root.textContent ?? "";
        expect(text).not.toContain("Project skills");
        expect(text).not.toContain("Project tools");
        expect(text).not.toContain("this project defines no");
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

    it("shows orchestrator tool controls without the removed System section", () => {
        // A project agent (global null) has NO System section.
        renderAgentSettings(root, data(), deps());
        let text = root.textContent ?? "";
        expect(text).not.toContain("System");
        // The orchestrator (global present) shows writable tools, not System.
        renderAgentSettings(
            root,
            data({
                agent: agent({ id: "orchestrator", name: "Orchestrator" }),
                global: globalSections(),
                mcpServers: [
                    {
                        id: "scufris",
                        status: "ok",
                        detail: "1 tool",
                        tools: [
                            {
                                name: "host_stats",
                                description: "host",
                                server: "scufris",
                                args: [],
                                parameters: [],
                                enabled: true,
                                available: true,
                            },
                        ],
                    },
                ],
            }),
            deps(),
        );
        text = root.textContent ?? "";
        expect(text).not.toContain("System");
        expect(
            root.querySelector('.settings__toggle[aria-label="enabled"]'),
        ).toBeNull();
        expect(
            root.querySelector('.settings__toggle[aria-label="tools"]'),
        ).toBeNull();
        expect(text).not.toContain("auth mode");
        expect(text).not.toContain("sandbox");
        // The removed "MCP servers" management card and "Profiles" switcher.
        expect(text).not.toContain("MCP servers");
        expect(text).not.toContain("Profiles");
        expect(text).toContain("MCP tools"); // the grouped tools section
        expect(text).toContain("scufris"); // the MCP tools server block header
        expect(text).toContain("host_stats"); // the tools catalog
        // Writable console -> the tool has an enable toggle.
        expect(root.querySelector(".tool-card__toggle")).not.toBeNull();
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
        // No writable global sections on a read-only server.
        expect(root.textContent).not.toContain("System");
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
