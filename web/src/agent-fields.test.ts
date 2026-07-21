import { describe, expect, it } from "vitest";

import { agentFields } from "./agent-fields";

describe("agentFields", () => {
    it("defaults to codex/manual and empty text for a fresh create", () => {
        const f = agentFields("new agent");
        expect(f.name.value).toBe("");
        expect(f.backend.value).toBe("codex");
        expect(f.description.value).toBe("");
        expect(f.mode.value).toBe("manual");
        expect(f.read()).toEqual({
            name: "",
            backend: "codex",
            description: "",
            permission_mode: "manual",
        });
    });

    it("prefills from initial values", () => {
        const f = agentFields("agent settings", {
            name: "Builder",
            backend: "claude",
            description: "does things",
            permission_mode: "auto",
        });
        expect(f.name.value).toBe("Builder");
        expect(f.backend.value).toBe("claude");
        expect(f.description.value).toBe("does things");
        expect(f.mode.value).toBe("auto");
    });

    it("labels the backend options with friendly names and prefixes aria labels", () => {
        const f = agentFields("agent settings");
        expect(f.name.getAttribute("aria-label")).toBe("agent settings name");
        expect(f.backend.getAttribute("aria-label")).toBe(
            "agent settings backend",
        );
        // Friendly labels, not raw ids.
        expect(f.backend.textContent).toContain("Codex");
        expect(f.backend.textContent).toContain("Claude");
        expect(f.backend.textContent).not.toContain("claude");
    });

    it("trims name and description in read()", () => {
        const f = agentFields("new agent");
        f.name.value = "  Spaced  ";
        f.description.value = "  trimmed  ";
        expect(f.read()).toEqual({
            name: "Spaced",
            backend: "codex",
            description: "trimmed",
            permission_mode: "manual",
        });
    });
});
