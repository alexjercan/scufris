// What already happened: the decided cards (with their live apply output and the
// undo they may offer) and the root helper's own audit table.

import { el } from "./common";
import type { HostActionRecord, HostAuditRecord } from "./host-types";
import { dispatch, type HostActions } from "./host-actions";
import { button, formatRequester, line, riskBadge, text } from "./host-format";

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

export function decidedCard(
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

// --- the audit table --------------------------------------------------------

export function auditTable(
    rows: HostAuditRecord[],
    failed: string,
): HTMLElement {
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
