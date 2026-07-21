// The shared agent field controls (name, backend, description, permission mode)
// used by BOTH the create form (agents-view) and the per-agent settings-edit
// form (agent-detail-view). Keeping one builder means the two forms cannot drift
// in their inputs, labels or defaults. Side-effect-free so jsdom tests drive it.

import { AGENT_BACKENDS, PERMISSION_MODES, backendLabel } from "./common";

// The values these controls collect. The create form adds `project_id` on top;
// the settings form PATCHes exactly this shape.
export interface AgentFieldValues {
    name: string;
    backend: string;
    description: string;
    permission_mode: string;
}

// The built controls plus a `read()` that returns their trimmed current values.
export interface AgentFields {
    name: HTMLInputElement;
    backend: HTMLSelectElement;
    description: HTMLTextAreaElement;
    mode: HTMLSelectElement;
    read(): AgentFieldValues;
}

function select(
    ariaLabel: string,
    options: readonly string[],
    toLabel: (value: string) => string,
): HTMLSelectElement {
    const sel = document.createElement("select");
    sel.className = "settings__input";
    sel.setAttribute("aria-label", ariaLabel);
    for (const value of options) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = toLabel(value);
        sel.appendChild(opt);
    }
    return sel;
}

// Build the shared controls. `context` prefixes each aria-label
// ("new agent" for create, "agent settings" for the settings form); `initial`
// prefills the current values (empty for a fresh create).
export function agentFields(
    context: string,
    initial: Partial<AgentFieldValues> = {},
): AgentFields {
    const name = document.createElement("input");
    name.type = "text";
    name.placeholder = "name";
    name.className = "settings__input";
    name.setAttribute("aria-label", `${context} name`);
    name.value = initial.name ?? "";

    const backend = select(`${context} backend`, AGENT_BACKENDS, backendLabel);
    backend.value = initial.backend ?? AGENT_BACKENDS[0];

    const description = document.createElement("textarea");
    description.placeholder = "description (optional)";
    description.className = "settings__input";
    description.setAttribute("aria-label", `${context} description`);
    description.value = initial.description ?? "";

    const mode = select(
        `${context} permission mode`,
        PERMISSION_MODES,
        (value) => value,
    );
    mode.value = initial.permission_mode ?? "manual";

    return {
        name,
        backend,
        description,
        mode,
        read: () => ({
            name: name.value.trim(),
            backend: backend.value,
            description: description.value.trim(),
            permission_mode: mode.value,
        }),
    };
}
