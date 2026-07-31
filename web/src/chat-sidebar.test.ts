import { beforeEach, describe, expect, it, vi } from "vitest";

import type { SessionContext, SessionInfo, UsageQuota } from "./agent-types";
import { renderContext, renderSessions, renderUsage } from "./chat-sidebar";

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
        primary: { used_percent: 34, window_minutes: 10080, resets_at: null },
        secondary: null,
        ...over,
    };
}

const noopActions = { onOpen: () => undefined, onDelete: () => undefined };

beforeEach(() => {
    document.body.innerHTML =
        '<div id="session-list"></div>' +
        '<div id="context-panel"></div><div id="usage-meter"></div>';
});

describe("renderSessions", () => {
    it("lists sessions, highlights the current, and wires open/delete", () => {
        const onOpen = vi.fn();
        const onDelete = vi.fn();
        renderSessions(
            [
                session({ id: "s1", title: "first" }),
                session({ id: "s2", title: "second" }),
            ],
            "s2",
            { onOpen, onDelete },
        );
        const items = document.querySelectorAll("#session-list .session");
        expect(items.length).toBe(2);
        expect(items[0].textContent).toContain("first");
        expect(items[1].classList.contains("is-active")).toBe(true);
        items[0].querySelector<HTMLButtonElement>(".session__open")?.click();
        expect(onOpen).toHaveBeenCalledWith("s1");
        items[0].querySelector<HTMLButtonElement>(".session__del")?.click();
        expect(onDelete).toHaveBeenCalledWith("s1", "first");
    });

    it("shows an empty state and escapes hostile titles", () => {
        renderSessions([], null, noopActions);
        expect(document.getElementById("session-list")?.textContent).toContain(
            "no sessions",
        );
        renderSessions(
            [session({ title: "<img src=x onerror=alert(1)>" })],
            null,
            noopActions,
        );
        expect(document.querySelector("#session-list img")).toBeNull();
        expect(document.getElementById("session-list")?.textContent).toContain(
            "<img src=x onerror=alert(1)>",
        );
    });
});

describe("renderContext", () => {
    it("shows window usage, token mix, counts and a freshness hint", () => {
        renderContext(ctx());
        const panel = document.getElementById("context-panel");
        expect(panel?.hidden).toBe(false);
        const text = panel?.textContent ?? "";
        expect(text).toContain("this session");
        expect(text).toContain("6%");
        expect(text).toContain("3 / 2");
        expect(text).toContain("as of last turn");
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
    it("shows the account box: window, percent, plan and freshness", () => {
        renderUsage(quota());
        const meter = document.getElementById("usage-meter");
        expect(meter?.hidden).toBe(false);
        const text = meter?.textContent ?? "";
        expect(text).toContain("account");
        expect(text).toContain("weekly");
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
