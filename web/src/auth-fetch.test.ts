import { readdirSync, readFileSync } from "node:fs";
import { basename, resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { apiFetch, csrfToken, sendJson } from "./common";

// jsdom refuses real navigation; spy on assign so the redirect is observable.
const assign = vi.fn();

beforeEach(() => {
    assign.mockClear();
    Object.defineProperty(window, "location", {
        configurable: true,
        value: { pathname: "/agents/", search: "?x=1", assign },
    });
    document.cookie = "scufris_csrf=tok-123";
});

afterEach(() => {
    vi.unstubAllGlobals();
    document.cookie = "scufris_csrf=; max-age=0";
});

function stubFetch(status = 200): ReturnType<typeof vi.fn> {
    const spy = vi.fn().mockResolvedValue(new Response("{}", { status }));
    vi.stubGlobal("fetch", spy);
    return spy;
}

function headersOf(spy: ReturnType<typeof vi.fn>): Headers {
    return (spy.mock.calls[0][1] as RequestInit).headers as Headers;
}

describe("apiFetch", () => {
    it("reads the CSRF token from its cookie", () => {
        expect(csrfToken()).toBe("tok-123");
    });

    it("attaches the CSRF header to state-changing requests", async () => {
        const spy = stubFetch();
        await apiFetch("/api/agents", { method: "POST" });
        expect(headersOf(spy).get("X-Scufris-CSRF")).toBe("tok-123");
    });

    it("does not attach it to safe requests", async () => {
        const spy = stubFetch();
        await apiFetch("/api/stats");
        expect(headersOf(spy).get("X-Scufris-CSRF")).toBeNull();
    });

    it("sends cookies same-origin so the session actually travels", async () => {
        const spy = stubFetch();
        await apiFetch("/api/stats");
        expect((spy.mock.calls[0][1] as RequestInit).credentials).toBe(
            "same-origin",
        );
    });

    it("redirects to the login page on a 401, preserving where we were", async () => {
        stubFetch(401);
        await apiFetch("/api/stats");
        expect(assign).toHaveBeenCalledWith(
            "/login/?next=%2Fagents%2F%3Fx%3D1",
        );
    });

    it("does not bounce the login page to itself", async () => {
        Object.defineProperty(window, "location", {
            configurable: true,
            value: { pathname: "/login/", search: "", assign },
        });
        stubFetch(401);
        await apiFetch("/api/auth/session");
        expect(assign).not.toHaveBeenCalled();
    });

    it("routes sendJson through the same seam", async () => {
        const spy = stubFetch();
        await sendJson("/api/agents/x", "PATCH", { name: "n" });
        expect(headersOf(spy).get("X-Scufris-CSRF")).toBe("tok-123");
    });
});

describe("the frontend has one API seam", () => {
    // The backend gates every route in ONE middleware; the frontend must match it
    // with ONE fetch wrapper, or a new call site silently misses the CSRF header
    // and the 401 redirect. Guard it by source, recursively, and cover the other
    // ways a module can reach the network - not just `fetch(`.
    //
    // Allowed:
    //   common.ts    - defines the seam, so it makes the one real `fetch` call.
    //   login.ts     - has no session and no CSRF cookie yet (the server issues
    //                  both in that very response), and apiFetch's 401 handler
    //                  would bounce the login page to itself.
    //   chat-stream.ts - opens an `EventSource`, which cannot carry a custom
    //                  header at all. Same-origin EventSource does send cookies,
    //                  and a 401 closes the stream (settling the promise) rather
    //                  than looping, so it is safe - but it is a deliberate
    //                  exception, listed here so it stays a decision rather than
    //                  an oversight.
    const allowed: Record<string, string[]> = {
        "common.ts": ["fetch("],
        "login.ts": ["fetch("],
        "chat-stream.ts": ["new EventSource("],
    };
    const forbidden: [string, RegExp][] = [
        ["fetch(", /(?<![.\w])fetch\s*\(/],
        ["new EventSource(", /new\s+EventSource\s*\(/],
        ["new XMLHttpRequest(", /new\s+XMLHttpRequest\s*\(/],
        ["navigator.sendBeacon(", /navigator\.sendBeacon\s*\(/],
    ];

    function sources(dir: string): string[] {
        const out: string[] = [];
        for (const entry of readdirSync(dir, { withFileTypes: true })) {
            const full = resolve(dir, entry.name);
            if (entry.isDirectory()) {
                out.push(...sources(full));
            } else if (
                entry.name.endsWith(".ts") &&
                !entry.name.endsWith(".test.ts")
            ) {
                out.push(full);
            }
        }
        return out;
    }

    it("routes every network call through apiFetch", () => {
        const offenders: string[] = [];
        for (const file of sources(resolve("src"))) {
            const name = basename(file);
            // Strip comments so prose mentioning fetch does not trip this.
            const code = readFileSync(file, "utf8")
                .replace(/\/\*[\s\S]*?\*\//g, "")
                .replace(/^\s*\/\/.*$/gm, "");
            for (const [label, pattern] of forbidden) {
                if (!pattern.test(code)) continue;
                if ((allowed[name] ?? []).includes(label)) continue;
                offenders.push(`${name}: ${label}`);
            }
        }
        expect(offenders).toEqual([]);
    });

    it("actually walks a non-trivial number of modules", () => {
        // Guard the guard: a broken walk would pass vacuously.
        expect(sources(resolve("src")).length).toBeGreaterThan(20);
    });
});
