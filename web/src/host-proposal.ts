// The pending queue: one card per proposal waiting for a decision, with its
// commands, its preview, its undo sentence and its controls.
//
// The confirmation requirement is the BACKEND's answer, not this view's opinion.
// `record.confirmation` says what approving costs (its risk phrase, the undo
// sentence, and the acknowledgement token a one-way action needs); this file
// renders that. A one-way action gets NO ordinary approve control at all - the
// only button that can approve it is the one that sends the token - so the
// proportionate friction is structural, and it is enforced again by the service,
// which refuses an approve without it. Keeping every approve control in this one
// module is what makes that rule reviewable in one place.

import { el } from "./common";
import type {
    HostActionRecord,
    HostConfirmation,
    HostPreview,
    HostStep,
} from "./host-types";
import { dispatch, type HostActions } from "./host-actions";
import {
    button,
    expiryMillis,
    formatExpiry,
    formatRequester,
    line,
    riskBadge,
    staleReason,
    text,
} from "./host-format";

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

export function pendingCard(
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
