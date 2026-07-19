import { beforeEach, describe, expect, it } from "vitest";

import type {
    AgentInfo,
    AgentTool,
    ChatReply,
    SessionContext,
    SessionInfo,
    TokenUsage,
    UsageQuota,
} from "./common";
import {
    applyUsage,
    messageMeta,
    renderAgentPanel,
    renderContext,
    renderSessions,
    renderUsage,
    _resetAgentState,
} from "./agent-view";

const info: AgentInfo = {
    model: "gpt-5.5",
    auth_mode: "chatgpt",
    enabled: true,
};

function tool(name: string, description = "does a thing"): AgentTool {
    return { name, description };
}

function usage(input: number, output: number): TokenUsage {
    return {
        input_tokens: input,
        cached_input_tokens: 0,
        output_tokens: output,
        reasoning_output_tokens: 0,
    };
}

function reply(over: Partial<ChatReply> = {}): ChatReply {
    return { text: "hi", tool_calls: [], usage: null, ...over };
}

function session(over: Partial<SessionInfo> = {}): SessionInfo {
    return {
        id: "s1",
        title: "a session",
        started_at: null,
        updated_at: null,
        git_branch: null,
        cwd: null,
        ...over,
    };
}

function ctx(over: Partial<SessionContext> = {}): SessionContext {
    return {
        session_id: "s1",
        context_window: 258400,
        input_tokens: 14612,
        cached_input_tokens: 9984,
        output_tokens: 74,
        reasoning_output_tokens: 43,
        total_tokens: 14700,
        turn_count: 3,
        tool_call_count: 2,
        ...over,
    };
}

function quota(over: Partial<UsageQuota> = {}): UsageQuota {
    return {
        plan_type: "plus",
        primary: {
            used_percent: 34,
            window_minutes: 10080,
            resets_at: null,
        },
        secondary: null,
        ...over,
    };
}

beforeEach(() => {
    document.body.innerHTML =
        '<span id="agent-model"></span><span id="agent-usage"></span>' +
        '<button id="agent-tools-toggle" hidden></button>' +
        '<div id="agent-tools" hidden></div>' +
        '<div id="session-list"></div>' +
        '<div id="context-panel"></div><div id="usage-meter"></div>';
    _resetAgentState();
});

describe("renderContext", () => {
    it("shows window usage, token mix and turn/tool counts", () => {
        renderContext(ctx());
        const panel = document.getElementById("context-panel");
        expect(panel?.hidden).toBe(false);
        const text = panel?.textContent ?? "";
        expect(text).toContain("context");
        expect(text).toContain("6%"); // 14612 / 258400 ~ 5.66 -> 6%
        expect(text).toContain("3 / 2"); // turns / tools
        expect(panel?.querySelector(".bar__fill")).not.toBeNull();
    });

    it("hides when there is no active session", () => {
        renderContext(null);
        expect(document.getElementById("context-panel")?.hidden).toBe(true);
        renderContext(ctx({ context_window: 0 }));
        expect(document.getElementById("context-panel")?.hidden).toBe(true);
    });
});

describe("renderUsage", () => {
    it("shows the weekly window, percent and plan", () => {
        renderUsage(quota());
        const meter = document.getElementById("usage-meter");
        expect(meter?.hidden).toBe(false);
        const text = meter?.textContent ?? "";
        expect(text).toContain("weekly usage");
        expect(text).toContain("34%");
        expect(text).toContain("plus");
    });

    it("hides when there is no reported limit", () => {
        renderUsage(null);
        expect(document.getElementById("usage-meter")?.hidden).toBe(true);
        renderUsage(quota({ primary: null }));
        expect(document.getElementById("usage-meter")?.hidden).toBe(true);
    });
});

describe("renderSessions", () => {
    it("lists sessions and highlights the current one", () => {
        renderSessions(
            [
                session({ id: "s1", title: "first" }),
                session({ id: "s2", title: "second" }),
            ],
            "s2",
        );
        const items = document.querySelectorAll("#session-list .session");
        expect(items.length).toBe(2);
        expect(items[0].textContent).toContain("first");
        expect(items[1].classList.contains("is-active")).toBe(true);
        expect(items[0].classList.contains("is-active")).toBe(false);
    });

    it("shows an empty state when there are no sessions", () => {
        renderSessions([], null);
        expect(document.querySelector("#session-list .session")).toBeNull();
        expect(document.getElementById("session-list")?.textContent).toContain(
            "no sessions",
        );
    });

    it("does not inject markup from a hostile session title", () => {
        renderSessions(
            [session({ title: "<img src=x onerror=alert(1)>" })],
            null,
        );
        expect(document.querySelector("#session-list img")).toBeNull();
        expect(document.getElementById("session-list")?.textContent).toContain(
            "<img src=x onerror=alert(1)>",
        );
    });
});

describe("renderAgentPanel", () => {
    it("shows the model and lists the tools", () => {
        renderAgentPanel(info, [tool("host_stats"), tool("tatr_ls")]);
        expect(document.getElementById("agent-model")?.textContent).toContain(
            "gpt-5.5",
        );
        const toggle = document.getElementById("agent-tools-toggle");
        expect(toggle?.hasAttribute("hidden")).toBe(false);
        expect(toggle?.textContent).toBe("tools (2)");
        expect(
            document.querySelectorAll("#agent-tools .agent-tools__item").length,
        ).toBe(2);
        expect(document.getElementById("agent-tools")?.textContent).toContain(
            "host_stats",
        );
    });

    it("hides the toggle when there are no tools", () => {
        renderAgentPanel(info, []);
        expect(
            document
                .getElementById("agent-tools-toggle")
                ?.hasAttribute("hidden"),
        ).toBe(true);
    });

    it("does not inject markup from a hostile tool name", () => {
        renderAgentPanel(info, [tool("<img src=x onerror=alert(1)>")]);
        expect(document.querySelector("#agent-tools img")).toBeNull();
        expect(document.getElementById("agent-tools")?.textContent).toContain(
            "<img src=x onerror=alert(1)>",
        );
    });
});

describe("messageMeta", () => {
    it("renders tool chips and the token count", () => {
        const meta = messageMeta(
            reply({
                tool_calls: [
                    {
                        server: "scufris",
                        tool: "host_stats",
                        status: "completed",
                    },
                ],
                usage: usage(47000, 87),
            }),
        );
        expect(meta).not.toBeNull();
        expect(meta?.textContent).toContain("host_stats");
        expect(meta?.textContent).toContain("87 tok");
    });

    it("returns null when there are no tools or usage", () => {
        expect(messageMeta(reply())).toBeNull();
    });
});

describe("applyUsage", () => {
    it("accumulates output tokens and shows context, and resets", () => {
        applyUsage(usage(47000, 87));
        let text = document.getElementById("agent-usage")?.textContent ?? "";
        expect(text).toContain("ctx 47.0k");
        expect(text).toContain("87 out");

        applyUsage(usage(48000, 100));
        text = document.getElementById("agent-usage")?.textContent ?? "";
        expect(text).toContain("187 out"); // cumulative

        _resetAgentState();
        expect(document.getElementById("agent-usage")?.textContent).toBe("");
    });
});
