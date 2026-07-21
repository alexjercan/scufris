import { describe, expect, it } from "vitest";

import type { BackendOption } from "./common";
import { agentFields } from "./agent-fields";

function backends(): BackendOption[] {
    return [
        { id: "codex", label: "Codex", default_model: "gpt-5.5" },
        { id: "claude", label: "Claude", default_model: "claude-opus-4-8" },
    ];
}

describe("agentFields", () => {
    it("defaults to the first backend + its model, manual, empty text", () => {
        const f = agentFields("new agent", backends());
        expect(f.name.value).toBe("");
        expect(f.backend.value).toBe("codex");
        expect(f.model.value).toBe("gpt-5.5"); // codex default model
        expect(f.description.value).toBe("");
        expect(f.mode.value).toBe("manual");
        expect(f.read()).toEqual({
            name: "",
            backend: "codex",
            model: "gpt-5.5",
            description: "",
            permission_mode: "manual",
        });
    });

    it("prefills from initial values (incl. an explicit model)", () => {
        const f = agentFields("agent settings", backends(), {
            name: "Builder",
            backend: "claude",
            model: "claude-sonnet-4-6",
            description: "does things",
            permission_mode: "auto",
        });
        expect(f.name.value).toBe("Builder");
        expect(f.backend.value).toBe("claude");
        expect(f.model.value).toBe("claude-sonnet-4-6"); // the override, kept
        expect(f.description.value).toBe("does things");
        expect(f.mode.value).toBe("auto");
    });

    it("re-defaults the model when the backend changes", () => {
        const f = agentFields("agent settings", backends(), {
            backend: "codex",
            model: "gpt-5.5",
        });
        expect(f.model.value).toBe("gpt-5.5");
        f.backend.value = "claude";
        f.backend.dispatchEvent(new Event("change"));
        expect(f.model.value).toBe("claude-opus-4-8");
        // ...and back again.
        f.backend.value = "codex";
        f.backend.dispatchEvent(new Event("change"));
        expect(f.model.value).toBe("gpt-5.5");
    });

    it("builds the backend options from the server list with friendly labels", () => {
        const f = agentFields("agent settings", backends());
        expect(f.backend.getAttribute("aria-label")).toBe(
            "agent settings backend",
        );
        expect(f.backend.textContent).toContain("Codex");
        expect(f.backend.textContent).toContain("Claude");
        expect(f.backend.textContent).not.toContain("claude");
        // Only the offered backends are options.
        const values = [...f.backend.options].map((o) => o.value);
        expect(values).toEqual(["codex", "claude"]);
    });

    it("trims name/description/model in read()", () => {
        const f = agentFields("new agent", backends());
        f.name.value = "  Spaced  ";
        f.description.value = "  trimmed  ";
        f.model.value = "  my-model  ";
        expect(f.read()).toEqual({
            name: "Spaced",
            backend: "codex",
            model: "my-model",
            description: "trimmed",
            permission_mode: "manual",
        });
    });
});
