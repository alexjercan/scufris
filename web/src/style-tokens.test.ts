import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

// The shared stylesheet is the whole app's theme. A `var(--x)` that references a
// token no definition provides resolves to nothing (transparent / inherited),
// which is how the buttons silently lost their fill (`background: var(--bg)` with
// only `--bg-0`/`--bg-1` defined). This guards that class of bug: every custom
// property USED without a fallback must be DEFINED somewhere in the sheet.
// Vitest runs from the web/ package root.
const css = readFileSync(resolve("src/style.css"), "utf8");

// Strip comments so a `--x:` or `var(--x)` inside a comment is not counted.
const code = css.replace(/\/\*[\s\S]*?\*\//g, "");

function definedTokens(source: string): Set<string> {
    const defs = new Set<string>();
    // A custom-property definition: `--name:` at a declaration position.
    for (const m of source.matchAll(/(--[\w-]+)\s*:/g)) defs.add(m[1]);
    return defs;
}

// Only `var(--x)` WITHOUT a fallback is unsafe; `var(--x, fallback)` is fine even
// if --x is undefined, so those are intentionally excluded.
function usedWithoutFallback(source: string): Set<string> {
    const used = new Set<string>();
    for (const m of source.matchAll(/var\(\s*(--[\w-]+)\s*\)/g)) used.add(m[1]);
    return used;
}

describe("style.css custom-property integrity", () => {
    it("references no undefined token without a fallback", () => {
        const defined = definedTokens(code);
        const undefinedRefs = [...usedWithoutFallback(code)].filter(
            (name) => !defined.has(name),
        );
        expect(undefinedRefs).toEqual([]);
    });

    it("defines the core theme tokens", () => {
        const defined = definedTokens(code);
        for (const token of [
            "--bg-0",
            "--text",
            "--text-muted",
            "--cyan",
            "--yellow",
            "--green",
            "--red",
            "--border",
            "--radius",
            "--font-mono",
            "--font-body",
        ]) {
            expect(defined.has(token)).toBe(true);
        }
    });

    it("no longer references the undefined --bg token that ghosted the buttons", () => {
        expect(usedWithoutFallback(code).has("--bg")).toBe(false);
    });
});
