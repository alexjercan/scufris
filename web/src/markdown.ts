// A small, safe-by-construction markdown renderer for assistant replies.
//
// The model's output is UNTRUSTED. Rather than parse it as HTML and sanitize
// (marked -> DOMPurify), this NEVER parses HTML: it tokenizes markdown and builds
// the DOM with `createTextNode` for every text run plus a fixed whitelist of
// elements. There is no `innerHTML` of model output anywhere, so there is no XSS
// surface. Link hrefs are scheme-validated. Side-effect-free (no import-time
// work), so it is unit-testable under jsdom.
//
// Scope: fenced code (+ copy), inline code, bold, italic, links, headings,
// ordered/unordered lists, blockquotes, paragraphs. Tables and nested lists are
// intentionally out of scope for now (rendered as plain paragraphs/text).

const SAFE_HREF = /^(https?:\/\/|mailto:|\/|#)/i;

// Split a fenced code block opener like "```python" into its language (or "").
function fenceLang(line: string): string {
    return line.replace(/^```/, "").trim();
}

function makeCodeBlock(code: string, lang: string): HTMLElement {
    const wrap = document.createElement("div");
    wrap.className = "md__code";

    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "md__copy";
    copy.textContent = "copy";
    copy.addEventListener("click", () => {
        // Guarded: clipboard is absent in jsdom and on insecure origins.
        const clip = navigator.clipboard;
        if (!clip) return;
        void clip.writeText(code).then(
            () => {
                copy.textContent = "copied";
                setTimeout(() => (copy.textContent = "copy"), 1200);
            },
            () => undefined,
        );
    });

    const pre = document.createElement("pre");
    const codeEl = document.createElement("code");
    if (lang) codeEl.className = `lang-${lang.replace(/[^\w-]/g, "")}`;
    codeEl.textContent = code; // textContent: no markup can escape
    pre.appendChild(codeEl);

    wrap.appendChild(copy);
    wrap.appendChild(pre);
    return wrap;
}

// Render inline markdown (code / bold / italic / links) into a parent, appending
// text nodes and whitelisted elements only. Non-nested; good enough for replies.
const INLINE =
    /(`[^`]+`)|(\*\*[^*]+\*\*)|(__[^_]+__)|(\*[^*]+\*)|(_[^_]+_)|(\[[^\]]+\]\([^)\s]+\))/;

function appendInline(parent: HTMLElement, text: string): void {
    let rest = text;
    while (rest.length > 0) {
        const match = INLINE.exec(rest);
        if (!match) {
            parent.appendChild(document.createTextNode(rest));
            return;
        }
        const at = match.index;
        if (at > 0)
            parent.appendChild(document.createTextNode(rest.slice(0, at)));
        const token = match[0];
        parent.appendChild(inlineToken(token));
        rest = rest.slice(at + token.length);
    }
}

function inlineToken(token: string): Node {
    if (token.startsWith("`")) {
        const code = document.createElement("code");
        code.className = "md__inline-code";
        code.textContent = token.slice(1, -1);
        return code;
    }
    if (token.startsWith("**") || token.startsWith("__")) {
        const strong = document.createElement("strong");
        strong.textContent = token.slice(2, -2);
        return strong;
    }
    if (token.startsWith("*") || token.startsWith("_")) {
        const em = document.createElement("em");
        em.textContent = token.slice(1, -1);
        return em;
    }
    // Link: [text](url)
    const link = /^\[([^\]]+)\]\(([^)\s]+)\)$/.exec(token);
    if (link) {
        const [, label, url] = link;
        if (SAFE_HREF.test(url)) {
            const anchor = document.createElement("a");
            anchor.href = url;
            anchor.textContent = label;
            anchor.target = "_blank";
            anchor.rel = "noopener noreferrer";
            return anchor;
        }
        // Unsafe scheme (javascript:, data:, ...): render inert as plain text.
        return document.createTextNode(`${label} (${url})`);
    }
    return document.createTextNode(token);
}

function makeParagraph(lines: string[]): HTMLElement {
    const p = document.createElement("p");
    appendInline(p, lines.join(" "));
    return p;
}

function makeList(items: string[], ordered: boolean): HTMLElement {
    const list = document.createElement(ordered ? "ol" : "ul");
    for (const item of items) {
        const li = document.createElement("li");
        appendInline(li, item);
        list.appendChild(li);
    }
    return list;
}

const UL_ITEM = /^\s*[-*+]\s+(.*)$/;
const OL_ITEM = /^\s*\d+\.\s+(.*)$/;
const HEADING = /^(#{1,6})\s+(.*)$/;

export function renderMarkdown(text: string): HTMLElement {
    const root = document.createElement("div");
    root.className = "md";
    const lines = text.replace(/\r\n?/g, "\n").split("\n");

    let i = 0;
    let paragraph: string[] = [];
    const flushParagraph = (): void => {
        if (paragraph.length > 0) {
            root.appendChild(makeParagraph(paragraph));
            paragraph = [];
        }
    };

    while (i < lines.length) {
        const line = lines[i];

        // Fenced code block.
        if (line.startsWith("```")) {
            flushParagraph();
            const lang = fenceLang(line);
            const code: string[] = [];
            i += 1;
            while (i < lines.length && !lines[i].startsWith("```")) {
                code.push(lines[i]);
                i += 1;
            }
            i += 1; // skip the closing fence (or run off the end)
            root.appendChild(makeCodeBlock(code.join("\n"), lang));
            continue;
        }

        // Blank line: paragraph break.
        if (line.trim() === "") {
            flushParagraph();
            i += 1;
            continue;
        }

        // Heading.
        const heading = HEADING.exec(line);
        if (heading) {
            flushParagraph();
            const level = Math.min(3, heading[1].length); // cap at h3 visually
            const h = document.createElement(`h${level}`);
            h.className = "md__h";
            appendInline(h, heading[2]);
            root.appendChild(h);
            i += 1;
            continue;
        }

        // Blockquote (single-level).
        if (/^\s*>\s?/.test(line)) {
            flushParagraph();
            const quote: string[] = [];
            while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
                quote.push(lines[i].replace(/^\s*>\s?/, ""));
                i += 1;
            }
            const bq = document.createElement("blockquote");
            appendInline(bq, quote.join(" "));
            root.appendChild(bq);
            continue;
        }

        // Lists (unordered / ordered), consecutive matching lines.
        if (UL_ITEM.test(line) || OL_ITEM.test(line)) {
            flushParagraph();
            const ordered = OL_ITEM.test(line);
            const re = ordered ? OL_ITEM : UL_ITEM;
            const items: string[] = [];
            while (i < lines.length && re.test(lines[i])) {
                const m = re.exec(lines[i]);
                if (m) items.push(m[1]);
                i += 1;
            }
            root.appendChild(makeList(items, ordered));
            continue;
        }

        // Otherwise: accumulate into the current paragraph.
        paragraph.push(line);
        i += 1;
    }
    flushParagraph();
    return root;
}
