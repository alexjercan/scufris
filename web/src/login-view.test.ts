import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { initLogin, safeNext } from "./login-view";

const loginHtml = readFileSync(resolve("src/login.html"), "utf8");

function mountForm(): void {
    // Mount the real page markup, so a renamed id breaks this test rather than
    // silently disabling the form in production.
    document.body.innerHTML = loginHtml.replace(
        /[\s\S]*<body>|<\/body>[\s\S]*/g,
        "",
    );
}

function jsonResponse(status: number, body: unknown): Response {
    return new Response(JSON.stringify(body), {
        status,
        headers: { "Content-Type": "application/json" },
    });
}

function submit(): void {
    const form = document.getElementById("login-form") as HTMLFormElement;
    form.dispatchEvent(
        new Event("submit", { cancelable: true, bubbles: true }),
    );
}

describe("safeNext", () => {
    it("keeps a local path", () => {
        expect(safeNext("/agents/")).toBe("/agents/");
        expect(safeNext("/stats/?tab=disks")).toBe("/stats/?tab=disks");
    });

    it("refuses anything that could leave this origin", () => {
        // A protocol-relative URL is the classic open-redirect payload: it looks
        // like a path and navigates off-site.
        expect(safeNext("//evil.example/phish")).toBe("/");
        expect(safeNext("https://evil.example")).toBe("/");
        expect(safeNext("/\\evil.example")).toBe("/");
        expect(safeNext("/ok\\..\\evil")).toBe("/");
        expect(safeNext(null)).toBe("/");
        expect(safeNext("")).toBe("/");
    });
});

describe("initLogin", () => {
    beforeEach(() => {
        mountForm();
    });

    it("navigates to the sanitized next target on success", async () => {
        const navigate = vi.fn();
        const post = vi.fn().mockResolvedValue(jsonResponse(200, { ok: true }));
        window.history.replaceState({}, "", "/login/?next=%2Fagents%2F");

        initLogin({ post, navigate });
        (document.getElementById("password") as HTMLInputElement).value = "pw";
        submit();
        await vi.waitFor(() => expect(navigate).toHaveBeenCalled());

        expect(post).toHaveBeenCalledWith("pw");
        expect(navigate).toHaveBeenCalledWith("/agents/");
    });

    it("does not follow an off-site next target", async () => {
        const navigate = vi.fn();
        const post = vi.fn().mockResolvedValue(jsonResponse(200, { ok: true }));
        window.history.replaceState({}, "", "/login/?next=%2F%2Fevil.example");

        initLogin({ post, navigate });
        submit();
        await vi.waitFor(() => expect(navigate).toHaveBeenCalled());

        expect(navigate).toHaveBeenCalledWith("/");
    });

    it("shows a message and clears the field on a wrong password", async () => {
        const navigate = vi.fn();
        const post = vi
            .fn()
            .mockResolvedValue(
                jsonResponse(401, { detail: "invalid credentials" }),
            );
        window.history.replaceState({}, "", "/login/");

        initLogin({ post, navigate });
        const input = document.getElementById("password") as HTMLInputElement;
        input.value = "wrong";
        submit();

        const error = document.getElementById("login-error") as HTMLElement;
        await vi.waitFor(() => expect(error.hidden).toBe(false));
        expect(error.textContent).toContain("Wrong password");
        expect(navigate).not.toHaveBeenCalled();
        // The password must not be left sitting in the DOM after a failure.
        expect(input.value).toBe("");
        // ...and the form stays usable for the next attempt.
        expect(
            (document.getElementById("login-submit") as HTMLButtonElement)
                .disabled,
        ).toBe(false);
    });

    it("surfaces the throttle message rather than a generic failure", async () => {
        const navigate = vi.fn();
        const post = vi.fn().mockResolvedValue(
            jsonResponse(429, {
                detail: "too many failed attempts; try again later",
            }),
        );
        window.history.replaceState({}, "", "/login/");

        initLogin({ post, navigate });
        submit();

        const error = document.getElementById("login-error") as HTMLElement;
        await vi.waitFor(() => expect(error.hidden).toBe(false));
        expect(error.textContent).toContain("too many failed attempts");
    });

    it("reports an unreachable server instead of hanging", async () => {
        const navigate = vi.fn();
        const post = vi.fn().mockRejectedValue(new Error("network"));
        window.history.replaceState({}, "", "/login/");

        initLogin({ post, navigate });
        submit();

        const error = document.getElementById("login-error") as HTMLElement;
        await vi.waitFor(() => expect(error.hidden).toBe(false));
        expect(error.textContent).toContain("Could not reach the server");
    });
});
