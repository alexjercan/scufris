import { describe, expect, it } from "vitest";

import { authLabel } from "./common";

describe("authLabel", () => {
    it("maps each backend's auth mode to a human label", () => {
        expect(authLabel("chatgpt")).toBe("ChatGPT");
        expect(authLabel("claude_ai")).toBe("claude.ai");
        expect(authLabel("api_key")).toBe("API key");
    });

    it("shows a dash for a null/empty mode (a backend with no login)", () => {
        expect(authLabel(null)).toBe("-");
        expect(authLabel("")).toBe("-");
    });

    it("passes through an unknown mode rather than hiding it", () => {
        expect(authLabel("something_new")).toBe("something_new");
    });
});
