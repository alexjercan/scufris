// Process view (below the stats cards): a collapsible, sortable table grouped by
// application. No import-time side effects (the `stats.ts` entry calls
// `startProcesses`), so `renderProcesses` is importable by the jsdom tests.

import {
    DEFAULT_POLL_SECONDS,
    el,
    escapeHtml,
    fetchJson,
    formatBytes,
    loadConfig,
} from "./common";

export interface ProcessInstance {
    pid: number;
    username: string;
    cpu_percent: number;
    mem_rss: number;
    num_threads: number;
    status: string;
}

export interface ProcessGroup {
    name: string;
    count: number;
    cpu_percent: number;
    mem_rss: number;
    instances: ProcessInstance[];
}

export interface ProcessList {
    groups: ProcessGroup[];
    total: number;
}

type SortKey = "cpu" | "mem";

// State persisted across re-renders (each poll re-renders, and expand/sort must
// survive that).
const _expanded = new Set<string>();
let _sortKey: SortKey = "cpu";
let _lastList: ProcessList | null = null;

// Test hook: reset the persisted expand/sort state between test cases.
export function _resetProcessState(): void {
    _expanded.clear();
    _sortKey = "cpu";
    _lastList = null;
}

function bySortKey(key: SortKey) {
    return (a: ProcessGroup, b: ProcessGroup): number =>
        key === "cpu" ? b.cpu_percent - a.cpu_percent : b.mem_rss - a.mem_rss;
}

function sortButton(key: SortKey, label: string): HTMLElement {
    const btn = el(
        "button",
        `proc__sortbtn ${_sortKey === key ? "is-active" : ""}`.trim(),
    );
    btn.textContent = label;
    btn.addEventListener("click", () => {
        _sortKey = key;
        if (_lastList) renderProcesses(_lastList);
    });
    return btn;
}

function groupRow(group: ProcessGroup, open: boolean): HTMLElement {
    const rowEl = el("div", `proc__group ${open ? "is-open" : ""}`.trim());
    rowEl.innerHTML =
        `<span class="proc__toggle">${open ? "▾" : "▸"}</span>` +
        `<span class="proc__name">${escapeHtml(group.name)}</span>` +
        `<span class="proc__count">${group.count}</span>` +
        `<span class="proc__cpu">${group.cpu_percent.toFixed(0)}%</span>` +
        `<span class="proc__mem">${formatBytes(group.mem_rss)}</span>`;
    rowEl.addEventListener("click", () => {
        if (_expanded.has(group.name)) _expanded.delete(group.name);
        else _expanded.add(group.name);
        if (_lastList) renderProcesses(_lastList);
    });
    return rowEl;
}

function instanceRow(inst: ProcessInstance): HTMLElement {
    const rowEl = el("div", "proc__inst");
    rowEl.innerHTML =
        `<span class="proc__toggle"></span>` +
        `<span class="proc__name">${inst.pid} ${escapeHtml(inst.username)}</span>` +
        `<span class="proc__count">${inst.num_threads}t</span>` +
        `<span class="proc__cpu">${inst.cpu_percent.toFixed(0)}%</span>` +
        `<span class="proc__mem">${formatBytes(inst.mem_rss)}</span>`;
    return rowEl;
}

export function renderProcesses(list: ProcessList): void {
    _lastList = list;
    const root = document.getElementById("processes");
    if (!root) return;
    root.replaceChildren();

    const head = el("div", "proc__head");
    head.appendChild(
        el(
            "h2",
            "card__title",
            `<span>Processes</span><span>${list.total} total</span>`,
        ),
    );
    const sort = el("div", "proc__sort");
    sort.appendChild(sortButton("cpu", "CPU"));
    sort.appendChild(sortButton("mem", "MEM"));
    head.appendChild(sort);
    root.appendChild(head);

    const table = el("div", "proc__table");
    const groups = [...list.groups].sort(bySortKey(_sortKey));
    for (const group of groups) {
        const open = _expanded.has(group.name);
        table.appendChild(groupRow(group, open));
        if (open) {
            for (const inst of group.instances)
                table.appendChild(instanceRow(inst));
        }
    }
    root.appendChild(table);
}

export async function startProcesses(): Promise<void> {
    const config = await loadConfig();
    const pollSeconds =
        config.poll_seconds > 0 ? config.poll_seconds : DEFAULT_POLL_SECONDS;
    const tick = (): void => {
        fetchJson<ProcessList>("/api/processes")
            .then(renderProcesses)
            .catch((err: unknown) => {
                console.error(err);
            });
    };
    tick();
    setInterval(tick, pollSeconds * 1000);
}
