import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

// The shared header partial (_header.html) is injected into every page by
// webpack, with `<%= basePath %>` interpolated to the deploy root (`/`). It is a
// template, not a rendered DOM node, so we assert against its source: the
// wordmark must be a link home. Vitest runs from the web/ package root.
const header = readFileSync(resolve("src/_header.html"), "utf8");

describe("_header.html", () => {
    it("makes the SCUFRIS wordmark a link to the landing root", () => {
        // The brand is an <a> to basePath (the landing / orchestrator chat),
        // not a plain div - clicking it returns to `/`.
        expect(header).toMatch(/<a class="brand" href="<%= basePath %>"[^>]*>/);
        expect(header).toContain("SCUFRIS");
        expect(header).toContain("scuffed jarvis");
    });
});
