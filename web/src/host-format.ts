// The text-only building blocks every host card is made of, and the read-only
// formatters beside them.
//
// Nothing host-derived reaches innerHTML on this page. A systemd unit is named by
// a FILE, and a preview quotes store paths, journal lines and command output, so
// every string is attacker-influenceable. `text()` and `line()` here are the ONLY
// ways text is set, and `el()` is called without its html argument throughout, so
// the page keeps no HTML sink to remember.

import { el } from "./common";
import type { HostActionRecord, HostConfirmation } from "./host-types";

export function text(
    tag: string,
    className: string,
    value: string,
): HTMLElement {
    const node = el(tag, className);
    node.textContent = value;
    return node;
}

// A key/value line (the shared `.row` shape used by the stats and agents cards).
export function line(key: string, value: string): HTMLElement {
    const row = el("div", "row");
    row.appendChild(text("span", "", key));
    row.appendChild(text("span", "", value));
    return row;
}

export function button(
    label: string,
    className: string,
    onClick: () => void,
): HTMLButtonElement {
    const node = document.createElement("button");
    node.type = "button";
    node.className = className;
    node.textContent = label;
    node.addEventListener("click", onClick);
    return node;
}

export function section(
    id: string,
    heading: string,
    body: HTMLElement[],
): HTMLElement {
    const node = el("section", "host__section");
    node.id = id;
    node.appendChild(text("h1", "host__heading", heading));
    for (const child of body) node.appendChild(child);
    return node;
}

// --- risk and expiry ---------------------------------------------------------

// A short label per risk class. The class also drives a CSS modifier, so a
// service restart and a system switch do not read identically - the requirement
// the task states, and the reason the badge carries both the letter and a word.
const RISK_WORD: Record<string, string> = {
    r1: "service",
    r2: "one-way",
    r3: "system",
};

export function riskBadge(confirmation: HostConfirmation): HTMLElement {
    const risk = confirmation.risk;
    const word = RISK_WORD[risk] ?? risk;
    const badge = text(
        "span",
        `host__risk host__risk--${/^[a-z0-9]+$/.test(risk) ? risk : "unknown"}`,
        `${risk.toUpperCase()} ${word}`,
    );
    badge.title = confirmation.risk_label;
    return badge;
}

// "in 7m 12s" / "expired". Whole seconds, because the operator is deciding against
// a window measured in minutes.
//
// BOTH arguments are milliseconds. The wire field is unix SECONDS, so the
// conversion happens once, at the call site that reads it (`expiryMillis`) - mixed
// units are one careless call away from a wrong countdown on the field that says
// how long the operator has.
export function formatExpiry(
    expiresAtMillis: number,
    nowMillis: number,
): string {
    const remaining = Math.round((expiresAtMillis - nowMillis) / 1000);
    if (remaining <= 0) return "expired";
    const mins = Math.floor(remaining / 60);
    const secs = remaining % 60;
    return mins > 0
        ? `in ${String(mins)}m ${String(secs)}s`
        : `in ${String(secs)}s`;
}

// The wire's unix-seconds expiry as milliseconds. One place converts.
export function expiryMillis(record: HostActionRecord): number {
    return record.proposal.expires_at * 1000;
}

// "3m" / "2h" / "4d" - how long ago something happened, for a row that is about
// recency rather than about a deadline.
export function formatAgo(atMillis: number, nowMillis: number): string {
    const seconds = Math.max(0, Math.round((nowMillis - atMillis) / 1000));
    if (seconds < 90) return `${String(seconds)}s`;
    if (seconds < 5400) return `${String(Math.round(seconds / 60))}m`;
    if (seconds < 172800) return `${String(Math.round(seconds / 3600))}h`;
    return `${String(Math.round(seconds / 86400))}d`;
}

// Who asked, from the record rather than from anything a caller supplied.
export function formatRequester(record: HostActionRecord): string {
    const requester = record.proposal.requester;
    const parts: string[] = [requester.actor || "unknown"];
    if (requester.agent) parts.push(`agent ${requester.agent}`);
    if (requester.run) parts.push(`run ${requester.run}`);
    return parts.join(" - ");
}

// Why this pending proposal can no longer be decided, or "" if it still can.
export function staleReason(record: HostActionRecord, now: number): string {
    const proposal = record.proposal;
    if (proposal.state === "drifted") {
        return (
            "This machine has changed since the preview was taken, so the preview " +
            "no longer describes what would happen. Ask for it again."
        );
    }
    if (proposal.state !== "pending") {
        return `The helper has already moved this proposal to ${proposal.state}.`;
    }
    if (expiryMillis(record) <= now) {
        return (
            "This proposal's window has closed, so the preview no longer describes " +
            "a decision that can be made. Ask for it again."
        );
    }
    return "";
}
