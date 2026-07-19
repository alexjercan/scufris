import { beforeEach, describe, expect, it } from "vitest";

import type { AgentInfo, AgentTool, ChatReply, TokenUsage } from "./common";
import {
    applyUsage,
    messageMeta,
    renderAgentPanel,
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

beforeEach(() => {
    document.body.innerHTML =
        '<span id="agent-model"></span><span id="agent-usage"></span>' +
        '<button id="agent-tools-toggle" hidden></button>' +
        '<div id="agent-tools" hidden></div>';
    _resetAgentState();
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
