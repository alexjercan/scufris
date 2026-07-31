// The shared pieces of the stats dashboard: the severity thresholds, the card and
// bar primitives, and the client-side rolling history behind the sparklines.
// Both card modules and `stats-view.ts` itself read from here.

import { el } from "./common";

export function formatUptime(seconds: number): string {
    const total = Math.floor(seconds);
    const days = Math.floor(total / 86400);
    const hours = Math.floor((total % 86400) / 3600);
    const mins = Math.floor((total % 3600) / 60);
    const parts: string[] = [];
    if (days > 0) parts.push(`${days}d`);
    if (hours > 0 || days > 0) parts.push(`${hours}h`);
    parts.push(`${mins}m`);
    return parts.join(" ");
}

export function severity(percent: number): string {
    if (percent >= 90) return "is-crit";
    if (percent >= 75) return "is-warn";
    return "";
}

export function bar(percent: number): HTMLElement {
    const wrap = el("div", "bar");
    const fill = el("div", `bar__fill ${severity(percent)}`.trim());
    fill.style.width = `${Math.max(0, Math.min(100, percent)).toFixed(1)}%`;
    wrap.appendChild(fill);
    return wrap;
}

const SVG_NS = "http://www.w3.org/2000/svg";

// Client-side rolling history: each poll appends one sample per series into a
// bounded array, so the sparklines fill from page-load the way btop's graphs
// fill from start (no backend history needed - /api/stats already carries every
// value graphed). Keyed by series id (cpu, load, mem, disk, net, gpu:<i>).
// Module state, so a test-reset hook is exported.
const HISTORY_LEN = 60;
const _history = new Map<string, number[]>();

export function pushHistory(key: string, value: number): number[] {
    const arr = _history.get(key) ?? [];
    arr.push(value);
    while (arr.length > HISTORY_LEN) arr.shift();
    _history.set(key, arr);
    return arr;
}

export function _resetStatsHistory(): void {
    _history.clear();
}

// A btop-style mini graph: a filled area under a polyline, drawn as inline SVG
// on a fixed 100x30 viewBox and stretched to the card width by CSS
// (preserveAspectRatio=none). Percent series pass max=100; rate/load series
// autoscale to the window max (floored at 1 to avoid divide-by-zero). Empty-safe
// and pure (holds no state), so it is unit-testable in jsdom.
export function sparkline(
    values: number[],
    max?: number,
    sevClass = "",
    title = "",
): SVGElement {
    const w = 100;
    const h = 30;
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", `spark ${sevClass}`.trim());
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svg.setAttribute("preserveAspectRatio", "none");
    // An SVG <title> is the native hover tooltip; it also names the graph for
    // assistive tech. Rendered even with no data so an empty graph still hints.
    if (title) {
        const t = document.createElementNS(SVG_NS, "title");
        t.textContent = title;
        svg.appendChild(t);
    }
    if (values.length === 0) return svg;
    const ceil = Math.max(max ?? Math.max(...values), 1);
    const n = values.length;
    const xAt = (i: number): number => (n === 1 ? w : (i / (n - 1)) * w);
    const yAt = (v: number): number =>
        h - (Math.max(0, Math.min(ceil, v)) / ceil) * h;
    const pts = values.map(
        (v, i) => `${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`,
    );
    const area = document.createElementNS(SVG_NS, "polygon");
    area.setAttribute("class", "spark__area");
    area.setAttribute("points", `0,${h} ${pts.join(" ")} ${w},${h}`);
    svg.appendChild(area);
    const line = document.createElementNS(SVG_NS, "polyline");
    line.setAttribute("class", "spark__line");
    line.setAttribute("points", pts.join(" "));
    svg.appendChild(line);
    return svg;
}

// A sparkline with a btop-style corner caption. `label` is the short visible
// tag (e.g. "cpu %"); `title` is the fuller hover tooltip. Returns a positioned
// wrapper so the caption floats over the graph without changing its height.
export function labeledSpark(
    label: string,
    title: string,
    values: number[],
    max?: number,
    sevClass = "",
): HTMLElement {
    const wrap = el("div", "spark-wrap");
    const caption = el("span", "spark__label");
    caption.textContent = label;
    wrap.appendChild(caption);
    wrap.appendChild(sparkline(values, max, sevClass, title));
    return wrap;
}

export function card(
    title: string,
    extra: string,
    body: (root: HTMLElement) => void,
): HTMLElement {
    const root = el("section", "card");
    root.appendChild(
        el("h2", "card__title", `<span>${title}</span><span>${extra}</span>`),
    );
    body(root);
    return root;
}

export function tempSeverity(temp: number): string {
    if (temp >= 85) return "is-crit";
    if (temp >= 70) return "is-warn";
    return "";
}
