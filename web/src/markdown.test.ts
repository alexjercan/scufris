import { describe, expect, it } from "vitest";

import { renderMarkdown } from "./markdown";

describe("renderMarkdown", () => {
    it("renders a fenced code block with the code and a copy button", () => {
        const md = renderMarkdown("here:\n\n```python\nprint('hi')\n```\n");
        const code = md.querySelector(".md__code pre code");
        expect(code?.textContent).toBe("print('hi')");
        expect(code?.className).toBe("lang-python");
        expect(md.querySelector(".md__copy")).not.toBeNull();
    });

    it("renders inline code, bold and italic", () => {
        const md = renderMarkdown("use `ls` and **run** it *now*");
        expect(md.querySelector(".md__inline-code")?.textContent).toBe("ls");
        expect(md.querySelector("strong")?.textContent).toBe("run");
        expect(md.querySelector("em")?.textContent).toBe("now");
    });

    it("renders unordered and ordered lists", () => {
        const ul = renderMarkdown("- one\n- two\n");
        expect(ul.querySelectorAll("ul li").length).toBe(2);
        const ol = renderMarkdown("1. first\n2. second\n");
        expect(ol.querySelectorAll("ol li").length).toBe(2);
    });

    it("renders a safe link with an href and blank target", () => {
        const md = renderMarkdown("see [docs](https://example.com/x)");
        const a = md.querySelector("a");
        expect(a?.getAttribute("href")).toBe("https://example.com/x");
        expect(a?.getAttribute("rel")).toContain("noopener");
        expect(a?.textContent).toBe("docs");
    });

    it("renders headings", () => {
        const md = renderMarkdown("# Title\n\nbody\n");
        expect(md.querySelector("h1.md__h")?.textContent).toBe("Title");
    });

    // --- security: model output is untrusted ---

    it("does not create markup from raw HTML in the reply", () => {
        const md = renderMarkdown("hello <img src=x onerror=alert(1)> world");
        expect(md.querySelector("img")).toBeNull();
        // The angle brackets survive as literal text.
        expect(md.textContent).toContain("<img src=x onerror=alert(1)>");
    });

    it("does not create a script tag from a fenced block", () => {
        const md = renderMarkdown("```\n<script>alert(1)</script>\n```");
        expect(md.querySelector("script")).toBeNull();
        expect(md.querySelector("code")?.textContent).toBe(
            "<script>alert(1)</script>",
        );
    });

    it("renders a javascript: link inert (no anchor)", () => {
        const md = renderMarkdown("[click](javascript:alert(1))");
        expect(md.querySelector("a")).toBeNull();
        // Falls back to plain text so nothing is clickable.
        expect(md.textContent).toContain("click");
        expect(md.textContent).toContain("javascript:alert(1)");
    });

    it("renders plain prose as paragraphs", () => {
        const md = renderMarkdown("first para\n\nsecond para");
        const paras = md.querySelectorAll("p");
        expect(paras.length).toBe(2);
        expect(paras[0].textContent).toBe("first para");
    });
});
