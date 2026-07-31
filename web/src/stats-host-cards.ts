// The host inspection cards (/api/host/overview).
//
// These come from a SEPARATE, much slower poll than the live gauges: the
// endpoint runs subprocesses server-side (systemctl, nixos-rebuild), so folding
// it into the 2s stats poll would make the live dashboard hostage to a rebuild.
//
// Each card renders its report's availability rather than its data alone. A card
// that cannot be read shows the REASON; an empty-but-fine card says so in words
// ("no failed units"). A blank card would read as "all good" in exactly the case
// where nothing was checked at all - the same rule the backend package enforces.

// Host-card DOM built with textContent, NEVER innerHTML.
//
// `card()` and `row()` take HTML strings, so every caller has to remember
// to escapeHtml host-derived values - and the host cards forgot, which is how a
// systemd unit named `<img src=x onerror=...>.service` (any file the operator
// can drop in ~/.config/systemd/user/) became script in the dashboard. Nearly
// every string on these cards comes from the machine: unit names, mountpoints,
// sensor labels, generation dates.
//
// So the host cards use these two helpers instead. They build nodes and assign
// textContent, which has no markup interpretation at all - there is no escaping
// to forget, because there is no HTML sink. A hostile-input jsdom test pins it.

import { el, formatBytes } from "./common";
import {
    type Availability,
    type HostOverview,
    type UnitList,
} from "./stats-types";
import { bar, tempSeverity } from "./stats-elements";

function hostCard(
    title: string,
    extra: string,
    body: (root: HTMLElement) => void,
): HTMLElement {
    const root = el("section", "card");
    const heading = el("h2", "card__title");
    const left = document.createElement("span");
    left.textContent = title;
    const right = document.createElement("span");
    right.textContent = extra;
    heading.append(left, right);
    root.appendChild(heading);
    body(root);
    return root;
}

function hostRow(label: string, value: string): HTMLElement {
    const node = el("div", "row");
    const left = document.createElement("span");
    left.textContent = label;
    const right = document.createElement("span");
    right.textContent = value;
    node.append(left, right);
    return node;
}

// A card's big number. `unit` is the small trailing label ("%", "C", "current").
function hostValue(text: string, unit = "", severityClass = ""): HTMLElement {
    const node = el("div", `card__value ${severityClass}`.trim());
    node.textContent = text;
    if (unit) {
        const small = document.createElement("small");
        small.textContent = unit;
        node.appendChild(small);
    }
    return node;
}

// The message a report's availability contributes, or "" when it is fully fine.
function availabilityNote(available: Availability): string {
    if (!available.ok) return `unavailable: ${available.reason}`;
    return available.caveat ? available.caveat : "";
}

// A note line under a card's value. Rendered for BOTH the unavailable reason and
// the caveat, so a partial answer is as visible as a missing one.
function noteRow(available: Availability): HTMLElement | null {
    const text = availabilityNote(available);
    if (!text) return null;
    const node = el("div", available.ok ? "card__note" : "card__note is-error");
    node.textContent = text;
    return node;
}

export function failedUnitsCard(overview: HostOverview): HTMLElement {
    const scopes: [string, UnitList][] = [
        ["system", overview.failed_system_units],
        ["user", overview.failed_user_units],
    ];
    // The count is only trustworthy when EVERY scope was read AND neither was
    // capped. Showing one scope's total while the other failed renders "0
    // failed" over an unread scope, and showing a capped list's length states 50
    // as fact when there were 60 - both are the false reassurance these cards
    // exist to avoid. Either condition makes the number qualified, and the
    // per-scope rows say which scope is at fault.
    const complete = scopes.every(([, list]) => list.available.ok);
    const capped = scopes.some(([, list]) => list.truncated);
    const total = scopes.reduce((sum, [, list]) => sum + list.units.length, 0);
    const extra = !complete
        ? "partially unreadable"
        : `${String(total)}${capped ? "+" : ""} failed`;
    return hostCard("Failed units", extra, (root) => {
        root.appendChild(
            hostValue(
                complete ? `${String(total)}${capped ? "+" : ""}` : "?",
                "",
                complete && total > 0 ? "is-crit" : "",
            ),
        );
        const rows = el("div", "card__rows");
        for (const [scope, list] of scopes) {
            if (!list.available.ok) {
                rows.appendChild(hostRow(scope, "unreadable"));
                continue;
            }
            // A stable row per scope with a dash-equivalent ("none"), rather than
            // a section that appears and disappears between polls.
            const names = list.units.map((u) => u.name).join(", ");
            rows.appendChild(
                hostRow(
                    scope,
                    list.units.length === 0
                        ? "none"
                        : names + (list.truncated ? ", ..." : ""),
                ),
            );
        }
        root.appendChild(rows);
        if (capped) {
            const note = el("div", "card__note");
            note.textContent =
                "the failed-unit list was capped, so the count is a lower bound";
            root.appendChild(note);
        }
        for (const [, list] of scopes) {
            const note = noteRow(list.available);
            if (note) root.appendChild(note);
        }
    });
}

export function generationsCard(overview: HostOverview): HTMLElement {
    const report = overview.storage.generations;
    const generations = report.generations;
    const current = generations.find((g) => g.current);
    const extra = report.available.ok
        ? `${String(generations.length)} total`
        : "";
    return hostCard("Generations", extra, (root) => {
        root.appendChild(
            current
                ? hostValue(String(current.number), "current")
                : hostValue("?"),
        );
        const rows = el("div", "card__rows");
        if (current) {
            rows.appendChild(hostRow("built", current.date));
            rows.appendChild(hostRow("kernel", current.kernel_version || "-"));
        } else if (report.available.ok) {
            rows.appendChild(hostRow("current", "none reported"));
        }
        const previous = generations.filter((g) => !g.current).slice(0, 3);
        for (const gen of previous) {
            rows.appendChild(hostRow(String(gen.number), gen.date));
        }
        root.appendChild(rows);
        const note = noteRow(report.available);
        if (note) root.appendChild(note);
    });
}

export function nixStoreCard(overview: HostOverview): HTMLElement {
    const storage = overview.storage;
    const store = storage.nix_store;
    return hostCard("Nix store", store ? store.mountpoint : "", (root) => {
        root.appendChild(
            store
                ? hostValue(store.percent.toFixed(1), "% used")
                : hostValue("?"),
        );
        if (store) {
            root.appendChild(bar(store.percent));
            const rows = el("div", "card__rows");
            rows.appendChild(hostRow("free", formatBytes(store.free)));
            rows.appendChild(hostRow("used", formatBytes(store.used)));
            rows.appendChild(hostRow("total", formatBytes(store.total)));
            root.appendChild(rows);
        }
        const note = noteRow(
            store ? storage.filesystems.available : storage.available,
        );
        if (note) root.appendChild(note);
    });
}

export function thermalCard(overview: HostOverview): HTMLElement {
    const thermal = overview.thermal;
    const hottest = thermal.temperatures[0];
    const throttling = thermal.throttling;
    return hostCard("Thermal", hottest ? hottest.label : "", (root) => {
        root.appendChild(
            hottest
                ? hostValue(
                      hottest.celsius.toFixed(0),
                      "C",
                      tempSeverity(hottest.celsius),
                  )
                : hostValue("?"),
        );
        const rows = el("div", "card__rows");
        if (!throttling.available.ok) {
            // "Cannot tell" is not "it did not throttle" - the whole reason the
            // counters are shown rather than inferred from a temperature.
            rows.appendChild(hostRow("throttling", "unknown"));
        } else {
            const events = throttling.core_events + throttling.package_events;
            // Two rows, not one packed "N core / M package": they count
            // different things (one physical core vs the whole chip), and a
            // single row with two bare numbers reads as one quantity split in
            // two. The label carries the unit so the figure cannot be divided
            // by the wrong denominator.
            if (events === 0) {
                rows.appendChild(hostRow("throttled", "never since boot"));
            } else {
                rows.appendChild(
                    hostRow(
                        "core throttles",
                        // "on 3 of 16 cores", not "across 16 cores": only three
                        // cores actually threw events, and that concentration
                        // is the interesting part. "across 16" reads as a
                        // distribution over all of them.
                        `${String(throttling.core_events)} on ${String(
                            throttling.cores_throttled,
                        )} of ${String(throttling.cores_read)} cores`,
                    ),
                );
                rows.appendChild(
                    hostRow(
                        "package throttles",
                        `${String(throttling.package_events)} (whole chip)`,
                    ),
                );
            }
        }
        if (thermal.battery.available.ok && thermal.battery.present) {
            const percent = thermal.battery.percent;
            rows.appendChild(
                hostRow(
                    "battery",
                    percent === null ? "-" : `${percent.toFixed(0)}%`,
                ),
            );
        }
        root.appendChild(rows);
        const note =
            noteRow(thermal.available) ?? noteRow(throttling.available);
        if (note) root.appendChild(note);
    });
}
