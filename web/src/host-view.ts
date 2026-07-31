// Host actions page: the operator's approval queue and the record of what has
// been done to this machine.
//
// `renderHost` is PURE (no fetch, no timers) so the jsdom tests drive it
// directly; `startHost` does the polling, the live output stream and the API
// calls. No import-time side effects - the `host.ts` entry calls `startHost`.
//
// TWO RULES SHAPE THIS FILE.
//
// 1. Nothing host-derived reaches innerHTML. A systemd unit is named by a FILE,
//    and a preview quotes store paths, journal lines and command output, so every
//    string here is attacker-influenceable. The stored XSS that shipped in the
//    stats cards came from exactly this data
//    reaching `innerHTML`, so this module builds DOM with `textContent` only and
//    keeps no HTML sink to remember: `el()` is called WITHOUT its html argument
//    throughout, and `text()`/`line()` below are the only ways text is set.
//
// 2. The confirmation requirement is the BACKEND's answer, not this view's
//    opinion. `record.confirmation` says what approving costs (its risk phrase,
//    the undo sentence, and the acknowledgement token a one-way action needs);
//    this file renders that. A one-way action gets NO ordinary approve control at
//    all - the only button that can approve it is the one that sends the token -
//    so the proportionate friction is structural here and enforced again by the
//    service, which refuses an approve without it.

import {
    apiFetch,
    el,
    sendJson,
    type HostActionRecord,
    type HostAuditRecord,
    type HostConfirmation,
    type HostDigest,
    type HostDigestView,
    type HostPreview,
    type HostStep,
    type ScheduleState,
} from "./common";

// How often the queue is re-read. A human-paced surface: the operator is looking
// at a decision, not at a live gauge, and the backend reconciles with the helper
// on its own throttle behind this.
const POLL_SECONDS = 4;

// --- text-only building blocks ----------------------------------------------

function text(tag: string, className: string, value: string): HTMLElement {
    const node = el(tag, className);
    node.textContent = value;
    return node;
}

// A key/value line (the shared `.row` shape used by the stats and agents cards).
function line(key: string, value: string): HTMLElement {
    const row = el("div", "row");
    row.appendChild(text("span", "", key));
    row.appendChild(text("span", "", value));
    return row;
}

function button(
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

// --- the actions the page can take ------------------------------------------

// `startHost` wires these to the API; the jsdom tests pass fakes. Every mutating
// one resolves after the server applied it, and the caller reloads.
export interface HostActions {
    // Run one schedule's checks now. The server answers 202 and the run happens in
    // the background (a pass walks the nix store), so this resolves before the
    // digest exists - the poll is what shows it.
    runChecks(schedule: string): Promise<void>;
    approve(id: string, acknowledge: string): Promise<void>;
    deny(id: string, reason: string): Promise<void>;
    cancel(id: string): Promise<void>;
    revert(id: string): Promise<void>;
    reload(): Promise<void>;
}

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

async function dispatch(
    actions: HostActions,
    run: () => Promise<void>,
): Promise<void> {
    try {
        await run();
        // Cleared on success, or the banner outlives the failure it describes: a
        // refused approve followed by a successful deny went on reporting "409
        // already decided" forever, on the one page whose job is to say truthfully
        // what happened to this machine (review round 1, R1.2).
        lastError = "";
    } catch (err: unknown) {
        // Surfaced through the reload below (the page shows `error`), so a failed
        // decision is never silent - and never a dead end either: the reload
        // re-reads the real state, which for a 409 is the state the other surface
        // just created.
        lastError = err instanceof Error ? err.message : String(err);
    }
    await actions.reload();
}

// The last failed action's message. Module state so the render can show it after
// a reload; `_resetHostError` is the test-reset hook (the ledger's
// persistent-ui-state-needs-a-test-reset-hook).
let lastError = "";

export function _resetHostError(): void {
    lastError = "";
}

export function hostError(): string {
    return lastError;
}

// Whether the operator is mid-interaction inside the page.
//
// `renderHost` rebuilds everything, so a poll landing while someone is typing
// replaces the input under their hands: measured in jsdom, a partially typed
// acknowledgement token and the focus both vanished on the next render, which makes
// the one-way approve path - type the action's name, then press the button - a race
// the operator loses every four seconds (review round 1, R1.1).
export function isTyping(root: HTMLElement): boolean {
    const active = document.activeElement;
    if (!(active instanceof HTMLElement) || !root.contains(active))
        return false;
    return (
        active instanceof HTMLInputElement ||
        active instanceof HTMLTextAreaElement
    );
}

// --- risk, expiry and preview -----------------------------------------------

// A short label per risk class. The class also drives a CSS modifier, so a
// service restart and a system switch do not read identically - the requirement
// the task states, and the reason the badge carries both the letter and a word.
const RISK_WORD: Record<string, string> = {
    r1: "service",
    r2: "one-way",
    r3: "system",
};

function riskBadge(confirmation: HostConfirmation): HTMLElement {
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
// conversion happens once, at the call site that reads it (`expiryMillis`) - the
// first version took mixed units, which is one careless call away from a wrong
// countdown on the field that says how long the operator has (review round 1, R1.4).
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

function commands(steps: HostStep[]): HTMLElement {
    const wrap = el("div", "host__commands");
    wrap.appendChild(
        text(
            "h3",
            "card__subhead",
            steps.length > 1
                ? `${String(steps.length)} commands, in order`
                : "command",
        ),
    );
    const list = el("ol", "host__steps");
    for (const step of steps) {
        const item = el("li");
        item.appendChild(text("code", "host__argv", step.argv.join(" ")));
        // A multi-step action is never summarised into one line: each step keeps
        // its own label, because stopping between two of them is a state of its
        // own (for an activation: this boot and the next boot disagree).
        if (step.label)
            item.appendChild(text("span", "host__step-label", step.label));
        list.appendChild(item);
    }
    wrap.appendChild(list);
    return wrap;
}

function preview(view: HostPreview): HTMLElement {
    const wrap = el("section", "host__preview");
    wrap.appendChild(
        text("h3", "card__subhead", `preview (${view.kind}): ${view.label}`),
    );
    // The availability line is the honesty marker: it says when the preview could
    // not read something, and an empty one means "fully available".
    const availability = view.available;
    if (!availability.ok || availability.reason || availability.caveat) {
        const note = [availability.reason, availability.caveat]
            .filter((part) => part)
            .join(" - ");
        wrap.appendChild(
            text(
                "p",
                availability.ok ? "host__caveat" : "host__unavailable",
                note || "this preview could not be produced",
            ),
        );
    }
    if (view.lines.length > 0) {
        // A <pre> of the preview's own lines: they are formatted server-side (a
        // closure diff, a unit's state, a dependency list) and reflowing them
        // would destroy the alignment that makes them readable.
        wrap.appendChild(
            text("pre", "host__preview-body", view.lines.join("\n")),
        );
    } else {
        wrap.appendChild(
            text("p", "host__empty-line", "this action has no preview lines"),
        );
    }
    return wrap;
}

function undoLine(confirmation: HostConfirmation): HTMLElement {
    const wrap = el("p", confirmation.no_undo ? "host__no-undo" : "host__undo");
    wrap.appendChild(
        text("strong", "", confirmation.no_undo ? "NO UNDO: " : "UNDO: "),
    );
    wrap.appendChild(document.createTextNode(confirmation.undo));
    return wrap;
}

// --- the pending controls ---------------------------------------------------

function denyControls(
    record: HostActionRecord,
    actions: HostActions,
): HTMLElement {
    const id = record.proposal.id;
    const wrap = el("div", "host__deny");
    const reason = document.createElement("input");
    reason.type = "text";
    reason.className = "settings__input host__reason";
    reason.placeholder = "why not? (reaches the agent that asked)";
    reason.setAttribute("aria-label", `reason for denying ${id}`);
    wrap.appendChild(reason);
    wrap.appendChild(
        button("deny", "settings__btn host__btn-deny", () => {
            void dispatch(actions, () => actions.deny(id, reason.value.trim()));
        }),
    );
    return wrap;
}

// The approve control, in the ONE shape the action's confirmation allows.
//
// A reversible action gets a plain button (its undo sentence is shown above it,
// which is what makes that enough). A one-way action gets an input plus a button
// that stays disabled until the typed value matches the token the backend
// requires - and NO plain button exists on that card, so the ordinary path
// cannot approve it even by mistake. The service refuses a tokenless approve
// anyway; this is the surface half of the same rule.
function approveControls(
    record: HostActionRecord,
    actions: HostActions,
): HTMLElement {
    const id = record.proposal.id;
    const confirmation = record.confirmation;
    const wrap = el("div", "host__approve");
    if (confirmation.style !== "one_way") {
        wrap.appendChild(
            button("approve", "settings__btn host__btn-approve", () => {
                void dispatch(actions, () => actions.approve(id, ""));
            }),
        );
        return wrap;
    }

    wrap.classList.add("host__approve--one-way");
    wrap.appendChild(
        text(
            "p",
            "host__ack-prompt",
            `This cannot be undone. Type ${confirmation.acknowledge} to approve it.`,
        ),
    );
    const token = document.createElement("input");
    token.type = "text";
    token.className = "settings__input host__ack";
    token.placeholder = confirmation.acknowledge;
    token.setAttribute("aria-label", `acknowledge ${confirmation.acknowledge}`);
    const approve = button(
        "approve (cannot be undone)",
        "settings__btn settings__btn--danger host__btn-approve-one-way",
        () => {
            // Re-checked here as well as by the disabled attribute: a disabled
            // button is a UI state, and the value is what the backend verifies.
            if (token.value.trim() !== confirmation.acknowledge) return;
            void dispatch(actions, () =>
                actions.approve(id, confirmation.acknowledge),
            );
        },
    );
    approve.disabled = true;
    token.addEventListener("input", () => {
        approve.disabled = token.value.trim() !== confirmation.acknowledge;
    });
    wrap.appendChild(token);
    wrap.appendChild(approve);
    return wrap;
}

// --- one card per action ----------------------------------------------------

function pendingCard(
    record: HostActionRecord,
    actions: HostActions,
    now: number,
): HTMLElement {
    const proposal = record.proposal;
    const card = el("section", "card host__card host__card--pending");
    card.setAttribute("data-action-id", proposal.id);

    const title = el("h2", "card__title");
    title.appendChild(text("span", "", proposal.summary));
    title.appendChild(riskBadge(record.confirmation));
    card.appendChild(title);

    const rows = el("div", "card__rows");
    rows.appendChild(line("action", proposal.kind));
    rows.appendChild(line("asked by", formatRequester(record)));
    rows.appendChild(line("expires", formatExpiry(expiryMillis(record), now)));
    card.appendChild(rows);

    card.appendChild(commands(proposal.steps));
    card.appendChild(preview(proposal.preview));
    card.appendChild(undoLine(record.confirmation));

    // An expired or drifted proposal is not decidable: say so where the controls
    // would be, rather than offering a button the server will refuse.
    const stale = staleReason(record, now);
    if (stale) {
        card.appendChild(text("p", "host__stale", stale));
        return card;
    }

    const controls = el("div", "host__controls");
    controls.appendChild(approveControls(record, actions));
    controls.appendChild(denyControls(record, actions));
    card.appendChild(controls);
    return card;
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

function resultRows(record: HostActionRecord): HTMLElement {
    const rows = el("div", "card__rows");
    const result = record.result;
    if (result) {
        rows.appendChild(
            line(
                "result",
                result.ok ? "applied" : `FAILED (${result.outcome})`,
            ),
        );
        if (result.returncode !== null) {
            rows.appendChild(line("exit", String(result.returncode)));
        }
        if (result.steps_total > 1) {
            rows.appendChild(
                line(
                    "steps",
                    `${String(result.steps_completed)}/${String(result.steps_total)}`,
                ),
            );
        }
        if (result.detail) rows.appendChild(line("detail", result.detail));
        // A half-applied multi-step action is a state of its own, and the operator
        // must not read it as "it failed, so nothing happened".
        if (!result.ok && result.steps_completed > 0) {
            rows.appendChild(
                line(
                    "note",
                    "this action stopped part-way through; read the host before " +
                        "assuming nothing changed",
                ),
            );
        }
    }
    if (record.error) rows.appendChild(line("error", record.error));
    return rows;
}

function decidedCard(
    record: HostActionRecord,
    actions: HostActions,
    output: string,
): HTMLElement {
    const proposal = record.proposal;
    const card = el("section", "card host__card host__card--decided");
    card.setAttribute("data-action-id", proposal.id);

    const title = el("h2", "card__title");
    title.appendChild(text("span", "", proposal.summary));
    title.appendChild(riskBadge(record.confirmation));
    title.appendChild(
        text(
            "span",
            `host__decision host__decision--${record.decision}`,
            record.decision,
        ),
    );
    card.appendChild(title);

    const rows = el("div", "card__rows");
    rows.appendChild(line("action", proposal.kind));
    rows.appendChild(line("asked by", formatRequester(record)));
    rows.appendChild(line("decided by", record.decided_by || "-"));
    if (record.reason) rows.appendChild(line("reason", record.reason));
    card.appendChild(rows);
    card.appendChild(resultRows(record));

    // Live output while it runs, and the tail afterwards: an apply that takes a
    // minute (a system switch) must show progress rather than a spinner.
    const running =
        record.run_id !== null && record.result === null && !record.error;
    if (output) {
        card.appendChild(
            text("h3", "card__subhead", running ? "running" : "output"),
        );
        card.appendChild(text("pre", "host__output", output));
    } else if (running) {
        card.appendChild(text("p", "host__running", "running..."));
    }

    const controls = el("div", "host__controls");
    if (running) {
        controls.appendChild(
            button(
                "stop",
                "settings__btn settings__btn--danger host__btn-cancel",
                () => {
                    if (
                        !window.confirm(
                            "Stop this apply? Whatever it has already done still stands.",
                        )
                    ) {
                        return;
                    }
                    void dispatch(actions, () => actions.cancel(proposal.id));
                },
            ),
        );
    }
    // The undo is offered exactly where the RECORD says one exists - on an applied
    // action whose result carries a reversal - and it is itself a proposal: it gets
    // its own preview and its own approval, which is why the button says so.
    const reversal = record.result?.reversal;
    if (record.result?.ok && reversal?.possible) {
        const undo = el("div", "host__revert");
        undo.appendChild(text("p", "host__revert-summary", reversal.summary));
        undo.appendChild(
            button("propose the undo", "settings__btn host__btn-revert", () => {
                void dispatch(actions, () => actions.revert(proposal.id));
            }),
        );
        controls.appendChild(undo);
    }
    if (controls.childElementCount > 0) card.appendChild(controls);
    return card;
}

// --- the scheduled checks ---------------------------------------------------

function scheduleRow(state: ScheduleState, now: number): HTMLElement {
    const row = el("div", "row");
    row.appendChild(text("span", "", state.name));
    const when =
        state.last_run === null
            ? "never run"
            : `${formatAgo(state.last_run * 1000, now)} ago`;
    // The last RESULT, not just the time: `watch` says nothing when there is nothing
    // to say, so "did it fire" has to be answerable somewhere, and this is where.
    const summary = state.last_result
        ? `${when} - ${state.last_result}`
        : `${when} - next ${formatExpiry(state.next_due * 1000, now)}`;
    row.appendChild(text("span", "", summary));
    return row;
}

function digestCard(digest: HostDigest, now: number): HTMLElement {
    const card = el("section", "card host__card host__digest");
    const title = el("h2", "card__title");
    title.appendChild(
        text(
            "span",
            "",
            `${digest.schedule} - ${formatAgo(digest.at * 1000, now)} ago`,
        ),
    );
    title.appendChild(
        text(
            "span",
            `host__verdict host__verdict--${digest.verdict === "attention" ? "attention" : "ok"}`,
            digest.verdict,
        ),
    );
    card.appendChild(title);
    card.appendChild(text("pre", "host__digest-body", digest.text));
    if (!digest.delivered) {
        // A digest that was written but never reached the operator is exactly the
        // thing this page has to say out loud.
        card.appendChild(
            text(
                "p",
                "host__caveat",
                digest.delivery_error === "muted"
                    ? "not sent: delivery is muted"
                    : `not delivered: ${digest.delivery_error || "unknown reason"}`,
            ),
        );
    }
    return card;
}

function checksSection(
    data: HostDigestView | undefined,
    actions: HostActions,
    now: number,
): HTMLElement[] {
    if (data === undefined) {
        return [text("p", "host__empty", "reading the scheduled checks...")];
    }
    const body: HTMLElement[] = [];
    if (!data.enabled) {
        body.push(
            text(
                "p",
                "host__unconfigured",
                "the scheduled host checks are switched off (host_checks_enabled)",
            ),
        );
    }
    if (data.muted_until * 1000 > now) {
        body.push(
            text(
                "p",
                "host__caveat",
                "delivery is muted - the checks still run and are still recorded here",
            ),
        );
    }
    const rows = el("section", "card host__card");
    rows.appendChild(text("h2", "card__title", "schedules"));
    const list = el("div", "card__rows");
    for (const state of data.schedules)
        list.appendChild(scheduleRow(state, now));
    rows.appendChild(list);
    const controls = el("div", "host__controls");
    for (const state of data.schedules) {
        controls.appendChild(
            button(
                `run ${state.name} now`,
                "settings__btn host__btn-run",
                () => {
                    void dispatch(actions, () => actions.runChecks(state.name));
                },
            ),
        );
    }
    rows.appendChild(controls);
    body.push(rows);

    if (data.digests.length === 0) {
        body.push(
            text(
                "p",
                "host__empty",
                "no digest yet - the checks run on their own schedule, or press a button above",
            ),
        );
    } else {
        for (const digest of data.digests.slice(0, 5)) {
            body.push(digestCard(digest, now));
        }
    }
    return body;
}

// --- the audit table --------------------------------------------------------

function auditTable(rows: HostAuditRecord[], failed: string): HTMLElement {
    const wrap = el("div", "host__audit-wrap");
    if (failed) {
        wrap.appendChild(text("p", "host__unavailable", failed));
        return wrap;
    }
    if (rows.length === 0) {
        wrap.appendChild(
            text(
                "p",
                "host__empty",
                "the helper's log is empty: nothing has been asked of it yet",
            ),
        );
        return wrap;
    }
    const table = el("table", "host__audit");
    const head = el("tr");
    for (const heading of ["when", "event", "who", "what", "outcome"]) {
        head.appendChild(text("th", "", heading));
    }
    table.appendChild(head);
    for (const row of rows) {
        const tr = el("tr", `host__audit-row host__audit-row--${row.event}`);
        tr.appendChild(text("td", "", row.at));
        tr.appendChild(text("td", "", row.event));
        tr.appendChild(text("td", "", row.requester.actor || "-"));
        const what =
            row.steps.length > 0 ? row.steps[0].argv.join(" ") : row.detail;
        tr.appendChild(text("td", "", what || "-"));
        tr.appendChild(text("td", "", row.outcome || "-"));
        table.appendChild(tr);
    }
    wrap.appendChild(table);
    return wrap;
}

// --- the page ---------------------------------------------------------------

function section(
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
    // reads as fine (review round 1, R1.3).
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
    // is where that shows up, not vitest (`type-change-fails-strict-tsc-not-vitest`).
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
