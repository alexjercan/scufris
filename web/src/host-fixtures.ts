// Fixtures for the host page tests: one proposal, one record, one audit row,
// and a recording stand-in for the page's actions - shared by
// `host-view.test.ts` and `host-proposal.test.ts`.

import type {
    HostActionRecord,
    HostActionResult,
    HostAuditRecord,
    HostConfirmation,
    HostProposal,
} from "./host-types";
import type { HostViewData } from "./host-view";
import type { HostActions } from "./host-actions";

// A fixed "now" so expiry rendering is deterministic; the proposal below expires
// ten minutes after it.
export const NOW = 1_800_000_000_000;

export function confirmation(
    over: Partial<HostConfirmation> = {},
): HostConfirmation {
    return {
        style: "ordinary",
        risk: "r1",
        risk_label: "service control - changes a unit's runtime state",
        undo: "stop nginx.service, restoring it to inactive",
        no_undo: false,
        acknowledge: "",
        ...over,
    };
}

export function proposal(over: Partial<HostProposal> = {}): HostProposal {
    return {
        id: "act-1",
        kind: "unit_restart",
        risk: "r1",
        args: { unit: "nginx" },
        steps: [
            {
                argv: ["systemctl", "restart", "--", "nginx.service"],
                label: "",
            },
        ],
        summary: "restart the nginx.service unit",
        preview: {
            kind: "state",
            headline: "restart nginx.service",
            label: "systemd cannot simulate this; these lines are current state",
            available: { ok: true, reason: "", caveat: "" },
            lines: ["unit: nginx.service (Nginx)", "now:  active (running)"],
        },
        reversal: {
            possible: true,
            summary: "stop nginx.service, restoring it to inactive",
            kind: "unit_stop",
            args: { unit: "nginx" },
        },
        requester: { actor: "agent", agent: "host", run: "run-9" },
        created_at: NOW / 1000,
        expires_at: NOW / 1000 + 600,
        state: "pending",
        ...over,
    };
}

export function record(over: Partial<HostActionRecord> = {}): HostActionRecord {
    return {
        proposal: proposal(),
        decision: "pending",
        decided_by: "",
        decided_at: null,
        reason: "",
        run_id: null,
        result: null,
        error: "",
        confirmation: confirmation(),
        ...over,
    };
}

export function result(over: Partial<HostActionResult> = {}): HostActionResult {
    return {
        ok: true,
        outcome: "ok",
        returncode: 0,
        duration_seconds: 0.4,
        steps_completed: 1,
        steps_total: 1,
        reversal: {
            possible: true,
            summary: "stop nginx.service, restoring it to inactive",
            kind: "unit_stop",
            args: { unit: "nginx" },
        },
        detail: "",
        ...over,
    };
}

export function auditRow(over: Partial<HostAuditRecord> = {}): HostAuditRecord {
    return {
        ts: NOW / 1000,
        at: "2026-07-30T08:00:00+00:00",
        event: "applied",
        action_id: "act-1",
        kind: "unit_restart",
        risk: "r1",
        steps: [
            {
                argv: ["systemctl", "restart", "--", "nginx.service"],
                label: "",
            },
        ],
        requester: { actor: "operator:abc", agent: "host", run: "" },
        outcome: "ok",
        returncode: 0,
        duration_seconds: 0.4,
        steps_completed: 1,
        reversal: "stop nginx.service",
        restore_point: "active",
        detail: "",
        ...over,
    };
}

export function view(over: Partial<HostViewData> = {}): HostViewData {
    return {
        queue: [record()],
        audit: [auditRow()],
        configured: true,
        now: NOW,
        ...over,
    };
}

// A recording stand-in for the page's actions, so a test can assert WHAT the
// control sent rather than that something happened.
export interface RecordedActions extends HostActions {
    ran: string[];
    approved: [string, string][];
    denied: [string, string][];
    cancelled: string[];
    reverted: string[];
    reloads: () => number;
}

export function actions(): RecordedActions {
    const ran: string[] = [];
    const approved: [string, string][] = [];
    const denied: [string, string][] = [];
    const cancelled: string[] = [];
    const reverted: string[] = [];
    let reloads = 0;
    return {
        ran,
        approved,
        denied,
        cancelled,
        reverted,
        reloads: () => reloads,
        runChecks(schedule: string) {
            ran.push(schedule);
            return Promise.resolve();
        },
        approve(id: string, acknowledge: string) {
            approved.push([id, acknowledge]);
            return Promise.resolve();
        },
        deny(id: string, reason: string) {
            denied.push([id, reason]);
            return Promise.resolve();
        },
        cancel(id: string) {
            cancelled.push(id);
            return Promise.resolve();
        },
        revert(id: string) {
            reverted.push(id);
            return Promise.resolve();
        },
        reload() {
            reloads += 1;
            return Promise.resolve();
        },
    };
}

export function root(): HTMLElement {
    const node = document.createElement("main");
    document.body.replaceChildren(node);
    return node;
}

// A one-way (irreversible) proposal: the case whose approve control is gated.
export function oneWayFixture(): HostActionRecord {
    return record({
        proposal: proposal({
            id: "gc-1",
            kind: "gc_store",
            risk: "r2",
            summary: "delete every unreachable store path",
            steps: [{ argv: ["nix-store", "--gc"], label: "" }],
            reversal: {
                possible: false,
                summary: "ONE-WAY. Deleted store paths cannot be restored.",
                kind: null,
                args: {},
            },
        }),
        confirmation: confirmation({
            style: "one_way",
            risk: "r2",
            risk_label: "disposable cleanup - ONE-WAY",
            undo: "ONE-WAY. Deleted store paths cannot be restored.",
            no_undo: true,
            acknowledge: "gc_store",
        }),
    });
}
