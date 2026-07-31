// The scheduled host checks: what each schedule last found, and the digests it
// produced.

import { el } from "./common";
import type { HostDigest, HostDigestView, ScheduleState } from "./host-types";
import { dispatch, type HostActions } from "./host-actions";
import { button, formatAgo, formatExpiry, text } from "./host-format";

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

export function checksSection(
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
