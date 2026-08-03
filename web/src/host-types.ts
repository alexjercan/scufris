// --- privileged host actions ------------------------------------------------
//
// The wire shape of `/api/host/actions` and `/api/host/audit`. Mirrors the
// backend models (`scufris/host_actions.py`, `packages/hostd/`); the fields this
// surface renders are the ones the operator decides against, so the types are
// complete rather than trimmed to today's render.
//
// NOTE: `HostActionRecord` has no `id` of its own - the python `id` is a plain
// property and is not serialised. The id is `record.proposal.id`.

import type { Availability } from "./stats-types";

export interface HostStep {
    argv: string[];
    label: string;
}

export interface HostPreview {
    kind: string;
    headline: string;
    label: string;
    available: Availability;
    lines: string[];
}

export interface HostReversal {
    possible: boolean;
    summary: string;
    kind: string | null;
    args: Record<string, unknown>;
}

export interface HostRequester {
    actor: string;
    agent: string;
    run: string;
}

// What approving ONE action requires, computed by the backend so both approval
// surfaces render the same requirement instead of each inventing one.
export interface HostConfirmation {
    style: "ordinary" | "one_way";
    risk: string;
    risk_label: string;
    undo: string;
    no_undo: boolean;
    acknowledge: string;
}

export interface HostProposal {
    id: string;
    kind: string;
    risk: string;
    args: Record<string, unknown>;
    steps: HostStep[];
    summary: string;
    preview: HostPreview;
    reversal: HostReversal;
    requester: HostRequester;
    created_at: number;
    expires_at: number;
    state: string;
}

export interface HostActionResult {
    ok: boolean;
    outcome: string;
    returncode: number | null;
    duration_seconds: number;
    steps_completed: number;
    steps_total: number;
    reversal: HostReversal;
    detail: string;
}

export interface HostActionRecord {
    proposal: HostProposal;
    decision: "pending" | "approved" | "denied";
    decided_by: string;
    decided_at: number | null;
    reason: string;
    run_id: string | null;
    result: HostActionResult | null;
    error: string;
    confirmation: HostConfirmation;
}

// One line of the helper's own root-written log.
export interface HostAuditRecord {
    ts: number;
    at: string;
    event: string;
    action_id: string;
    kind: string | null;
    risk: string | null;
    steps: HostStep[];
    requester: HostRequester;
    outcome: string;
    returncode: number | null;
    duration_seconds: number;
    steps_completed: number;
    reversal: string;
    restore_point: string;
    detail: string;
}

// --- the scheduled host checks ----------------------------------------------

export interface ScheduleState {
    name: string;
    next_due: number;
    last_run: number | null;
    last_result: string;
    missed: number;
    runs: number;
}

export interface HostDigest {
    at: number;
    schedule: string;
    verdict: string;
    text: string;
    delivered: boolean;
    delivery_error: string;
    states: Record<string, string>;
}

export interface HostDigestView {
    schedules: ScheduleState[];
    digests: HostDigest[];
    muted_until: number;
    enabled: boolean;
}
