// Host actions page: the operator's approval queue and the record of what has
// been done to this machine.
//
// `renderHost` is PURE (no fetch, no timers) so the jsdom tests drive it
// directly; `startHost` does the polling, the live output stream and the API
// calls. No import-time side effects - the `host.ts` entry calls `startHost`.
//
// The sections live beside this file: the pending cards in `host-proposal.ts`,
// the decided cards and audit table in `host-history.ts`, the scheduled checks
// in `host-checks.ts`, and the text-only building blocks they all share in
// `host-format.ts`. `host-actions.ts` owns what the page can do.
//
// TWO RULES SHAPE THIS PAGE.
//
// 1. Nothing host-derived reaches innerHTML. A systemd unit is named by a FILE,
//    and a preview quotes store paths, journal lines and command output, so every
//    string here is attacker-influenceable. The stored XSS that shipped in the
//    stats cards came from exactly this data reaching `innerHTML`, so these
//    modules build DOM with `textContent` only and keep no HTML sink to remember:
//    `el()` is called WITHOUT its html argument throughout, and `host-format.ts`'s
//    `text()`/`line()` are the only ways text is set.
//
// 2. The confirmation requirement is the BACKEND's answer, not this view's
//    opinion. `record.confirmation` says what approving costs, and a one-way
//    action gets NO ordinary approve control at all - see `host-proposal.ts`,
//    which holds every approve control there is.

import { apiFetch, sendJson } from "./common";
import type {
    HostActionRecord,
    HostAuditRecord,
    HostDigestView,
} from "./host-types";
import { hostError, type HostActions } from "./host-actions";
import { section, text } from "./host-format";
import { pendingCard } from "./host-proposal";
import { auditTable, decidedCard } from "./host-history";
import { checksSection } from "./host-checks";

// How often the queue is re-read. A human-paced surface: the operator is looking
// at a decision, not at a live gauge, and the backend reconciles with the helper
// on its own throttle behind this.
const POLL_SECONDS = 4;

// What the page renders. `error` is the last failed call's message (a decision the
// OTHER surface just made arrives here as a 409, which is information rather than
// a fault); `configured` is false when the helper is not enabled on this host, in
// which case the queue is not broken - it does not exist.
export interface HostViewData {
    queue: HostActionRecord[];
    // The scheduled checks: what they found, and when each schedule last ran.
    // Undefined while the first poll is in flight.
    checks?: HostDigestView;
    audit: HostAuditRecord[];
    configured: boolean;
    // The server's own explanation of what is missing, when it gave one.
    unconfiguredDetail?: string;
    // Why the audit log could not be read, when it could not.
    auditFailed?: string;
    error?: string;
    // Live output of an apply, keyed by action id, appended as it streams.
    output?: Record<string, string>;
    // Injected in tests; defaults to the wall clock (expiry is relative).
    now?: number;
}

// Whether the operator is mid-interaction inside the page.
//
// `renderHost` rebuilds everything, so a poll landing while someone is typing
// replaces the input under their hands: measured in jsdom, a partially typed
// acknowledgement token and the focus both vanished on the next render, which makes
// the one-way approve path - type the action's name, then press the button - a race
// the operator loses every four seconds.
export function isTyping(root: HTMLElement): boolean {
    const active = document.activeElement;
    if (!(active instanceof HTMLElement) || !root.contains(active))
        return false;
    return (
        active instanceof HTMLInputElement ||
        active instanceof HTMLTextAreaElement
    );
}

// --- the page ---------------------------------------------------------------

export function renderHost(
    root: HTMLElement,
    data: HostViewData,
    actions: HostActions,
): void {
    const now = data.now ?? Date.now();
    const output = data.output ?? {};
    root.replaceChildren();

    const error = data.error ?? hostError();
    if (error) {
        // Shown, not swallowed: the most common one is a 409 because the other
        // surface (Telegram) just decided this action, which the operator should
        // read as news rather than as a failure.
        root.appendChild(text("p", "host__error", error));
    }

    if (!data.configured) {
        root.appendChild(
            text(
                "p",
                "host__unconfigured",
                "The privileged host helper is not configured on this machine, so " +
                    "there is nothing to approve - and nothing can be asked of it.",
            ),
        );
        if (data.unconfiguredDetail) {
            root.appendChild(
                text("p", "host__unconfigured", data.unconfiguredDetail),
            );
        }
        return;
    }

    const pending = data.queue.filter((r) => r.decision === "pending");
    const decided = data.queue.filter((r) => r.decision !== "pending");

    root.appendChild(
        section(
            "host-pending",
            pending.length > 0
                ? `Waiting for you (${String(pending.length)})`
                : "Waiting for you",
            pending.length > 0
                ? pending.map((record) => pendingCard(record, actions, now))
                : [
                      text(
                          "p",
                          "host__empty",
                          "nothing is waiting for a decision",
                      ),
                  ],
        ),
    );

    root.appendChild(
        section(
            "host-decided",
            "Recent decisions",
            decided.length > 0
                ? decided.map((record) =>
                      decidedCard(
                          record,
                          actions,
                          output[record.proposal.id] ?? "",
                      ),
                  )
                : [
                      text(
                          "p",
                          "host__empty",
                          "no action has been decided in this server's lifetime",
                      ),
                  ],
        ),
    );

    root.appendChild(
        section(
            "host-checks",
            "What has been watching",
            checksSection(data.checks, actions, now),
        ),
    );

    root.appendChild(
        section("host-audit", "The record", [
            text(
                "p",
                "host__audit-note",
                "Written by the root helper itself, so it holds actions this page " +
                    "never saw - including ones from before a restart.",
            ),
            auditTable(data.audit, data.auditFailed ?? ""),
        ]),
    );
}

// --- orchestration ----------------------------------------------------------

// Whether this box HAS a privileged helper, and the log if it does.
//
// The signal is the AUDIT endpoint, not the queue. Measured against a running
// server with no helper configured: `/api/host/actions` answers `200 []` (the
// app's own registry is simply empty), while `/api/host/audit` answers 503 with
// the sentence naming what to set. Reading "not configured" off the queue would
// therefore never fire, and the page would show an empty queue and an empty log -
// "nothing has been asked of it yet" - when the truth is that nothing CAN be.
interface AuditRead {
    configured: boolean;
    rows: HostAuditRecord[];
    detail: string;
    // Set when the read itself failed. "Could not be read" and "empty" are
    // different sentences, and rendering the first as the second is the blank that
    // reads as fine.
    failed: string;
}

async function readAudit(): Promise<AuditRead> {
    const resp = await apiFetch("/api/host/audit?limit=50");
    if (resp.status === 503) {
        // The server's own sentence names the env vars and the NixOS module, which
        // is more useful than anything this page could paraphrase.
        let detail = "";
        try {
            detail = ((await resp.json()) as { detail?: string }).detail ?? "";
        } catch {
            // no body; the page falls back to its own explanation
        }
        return { configured: false, rows: [], detail, failed: "" };
    }
    if (!resp.ok) {
        return {
            configured: true,
            rows: [],
            detail: "",
            failed: `the helper's log could not be read (HTTP ${String(resp.status)})`,
        };
    }
    return {
        configured: true,
        rows: (await resp.json()) as HostAuditRecord[],
        detail: "",
        failed: "",
    };
}

async function readChecks(): Promise<HostDigestView | undefined> {
    // Undefined rather than a throw: the scheduled checks are one section of this
    // page, and failing to read them must not cost the operator the approval queue
    // above them.
    const resp = await apiFetch("/api/host/digests");
    if (!resp.ok) return undefined;
    return (await resp.json()) as HostDigestView;
}

async function readQueue(): Promise<HostActionRecord[]> {
    const resp = await apiFetch("/api/host/actions");
    if (!resp.ok)
        throw new Error(`/api/host/actions -> ${String(resp.status)}`);
    return (await resp.json()) as HostActionRecord[];
}

export function startHost(): void {
    const found = document.getElementById("host");
    if (!found) return;
    // Bound to a non-null const: the guard above narrows `found`, but that
    // narrowing does not survive into the closures below (a nested function can be
    // called later, so tsc keeps the wider type) - and the webpack ts-loader build
    // is where that shows up, not vitest.
    const root: HTMLElement = found;

    // Live apply output, appended per action id as the SSE stream delivers it.
    const output: Record<string, string> = {};
    const streams = new Map<string, EventSource>();

    const actions: HostActions = {
        runChecks: async (schedule) => {
            await sendJson(
                `/api/host/digests/run?schedule=${encodeURIComponent(schedule)}`,
                "POST",
            );
        },
        approve: async (id, acknowledge) => {
            await sendJson(
                `/api/host/actions/${encodeURIComponent(id)}/approve`,
                "POST",
                acknowledge ? { acknowledge } : {},
            );
        },
        deny: async (id, reason) => {
            await sendJson(
                `/api/host/actions/${encodeURIComponent(id)}/deny`,
                "POST",
                { reason },
            );
        },
        cancel: async (id) => {
            await sendJson(
                `/api/host/actions/${encodeURIComponent(id)}/cancel`,
                "POST",
            );
        },
        revert: async (id) => {
            await sendJson(
                `/api/host/actions/${encodeURIComponent(id)}/revert`,
                "POST",
            );
        },
        reload: async () => {
            await refresh();
        },
    };

    // Attach to a running apply's output stream once, and drop it when the run
    // ends - the endpoint 404s when there is no live run, so a finished action
    // simply has no stream to keep.
    function follow(record: HostActionRecord): void {
        const id = record.proposal.id;
        const running =
            record.run_id !== null && record.result === null && !record.error;
        if (!running) {
            streams.get(id)?.close();
            streams.delete(id);
            return;
        }
        if (streams.has(id)) return;
        const source = new EventSource(
            `/api/host/actions/${encodeURIComponent(id)}/events`,
        );
        streams.set(id, source);
        source.addEventListener("message", (ev: MessageEvent<string>) => {
            try {
                const event = JSON.parse(ev.data) as {
                    type?: string;
                    text?: string;
                };
                if (event.type === "output" && event.text) {
                    output[id] = (output[id] ?? "") + event.text;
                    render();
                }
            } catch {
                // A frame this build cannot read is not worth breaking the page for.
            }
        });
        source.addEventListener("error", () => {
            source.close();
            streams.delete(id);
        });
    }

    let data: HostViewData = { queue: [], audit: [], configured: true, output };

    // A poll may hold its render back; a decision must not (its controls are gone
    // by then, and the operator is waiting to see what happened).
    function render(options: { poll?: boolean } = {}): void {
        data = { ...data, output, now: Date.now() };
        if (options.poll && isTyping(root)) return;
        renderHost(root, data, actions);
    }

    async function refresh(options: { poll?: boolean } = {}): Promise<void> {
        try {
            const [queue, audit, checks] = await Promise.all([
                readQueue(),
                readAudit(),
                readChecks(),
            ]);
            data = {
                queue,
                audit: audit.rows,
                checks,
                configured: audit.configured,
                unconfiguredDetail: audit.detail,
                auditFailed: audit.failed,
                output,
            };
            for (const record of queue) follow(record);
        } catch (err: unknown) {
            data = {
                ...data,
                error: err instanceof Error ? err.message : String(err),
            };
        }
        render(options);
    }

    void refresh();
    window.setInterval(() => {
        // The data is refreshed on every tick either way; only the RENDER waits for
        // the operator to stop typing, so nothing goes stale behind the deferral.
        void refresh({ poll: true });
    }, POLL_SECONDS * 1000);
}
