// The shared agent field controls (name, backend, model, description,
// permission mode) used by BOTH the create form (agents-view) and the per-agent
// settings-edit form (agent-detail-view). Keeping one builder means the two
// forms cannot drift in their inputs, labels or defaults. Side-effect-free so
// jsdom tests drive it.
//
// The backend options (and each backend's default model) come from the server
// (GET /api/agents/backends), so the picker is authoritative. Changing the
// backend auto-fills the model with that backend's default; the operator can
// still type an override before saving.

import { PERMISSION_MODES } from "./common";
import type { BackendOption } from "./agent-types";

// The values these controls collect. The create form adds `project_id` on top;
// the settings form PATCHes exactly this shape.
export interface AgentFieldValues {
    name: string;
    backend: string;
    model: string;
    description: string;
    permission_mode: string;
}

// The built controls plus a `read()` that returns their trimmed current values.
export interface AgentFields {
    name: HTMLInputElement;
    backend: HTMLSelectElement;
    model: HTMLInputElement;
    // The <datalist> of model suggestions the `model` input references (via its
    // `list` attribute); callers must append it to the DOM alongside `model`.
    modelList: HTMLDataListElement;
    description: HTMLTextAreaElement;
    mode: HTMLSelectElement;
    read(): AgentFieldValues;
}

function modeSelect(ariaLabel: string): HTMLSelectElement {
    const sel = document.createElement("select");
    sel.className = "settings__input";
    sel.setAttribute("aria-label", ariaLabel);
    for (const value of PERMISSION_MODES) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = value;
        sel.appendChild(opt);
    }
    return sel;
}

function backendSelect(
    ariaLabel: string,
    backends: BackendOption[],
): HTMLSelectElement {
    const sel = document.createElement("select");
    sel.className = "settings__input";
    sel.setAttribute("aria-label", ariaLabel);
    for (const b of backends) {
        const opt = document.createElement("option");
        opt.value = b.id;
        opt.textContent = b.label;
        sel.appendChild(opt);
    }
    return sel;
}

function textInput(ariaLabel: string, placeholder: string): HTMLInputElement {
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = placeholder;
    input.className = "settings__input";
    input.setAttribute("aria-label", ariaLabel);
    return input;
}

// Build the shared controls. `context` prefixes each aria-label
// ("new agent" for create, "agent settings" for the settings form). `backends`
// is the server's available-backend list (with default models); `initial`
// prefills the current values (empty for a fresh create).
export function agentFields(
    context: string,
    backends: BackendOption[],
    initial: Partial<AgentFieldValues> = {},
): AgentFields {
    const name = textInput(`${context} name`, "name");
    name.value = initial.name ?? "";

    const backend = backendSelect(`${context} backend`, backends);
    const backendById = (id: string): BackendOption | undefined =>
        backends.find((b) => b.id === id);
    const defaultModelFor = (id: string): string =>
        backendById(id)?.default_model ?? "";

    // The model field is a free-text input backed by a <datalist> of the selected
    // backend's models: a dropdown of suggestions that still accepts any typed id
    // (an operator can enter a model outside the built-in catalog).
    const model = textInput(`${context} model`, "model");
    const modelList = document.createElement("datalist");
    modelList.id = `agent-models-${context.replace(/\s+/g, "-")}`;
    model.setAttribute("list", modelList.id);
    const fillModelList = (id: string): void => {
        modelList.replaceChildren();
        for (const m of backendById(id)?.models ?? []) {
            const opt = document.createElement("option");
            opt.value = m;
            modelList.appendChild(opt);
        }
    };

    // Start on the initial backend (or the first offered), and prefill the model
    // from the agent or that backend's default; populate its suggestions.
    backend.value = initial.backend ?? backends[0]?.id ?? "";
    model.value = initial.model ?? defaultModelFor(backend.value);
    fillModelList(backend.value);

    // Switching the backend re-defaults the model AND swaps the suggestions, so a
    // claude agent never keeps a codex model or codex's suggestions. Free text
    // is still preserved (the operator can override).
    backend.addEventListener("change", () => {
        model.value = defaultModelFor(backend.value);
        fillModelList(backend.value);
    });

    const description = document.createElement("textarea");
    description.placeholder = "description (optional)";
    description.className = "settings__input";
    description.setAttribute("aria-label", `${context} description`);
    description.value = initial.description ?? "";

    const mode = modeSelect(`${context} permission mode`);
    mode.value = initial.permission_mode ?? "manual";

    return {
        name,
        backend,
        model,
        modelList,
        description,
        mode,
        read: () => ({
            name: name.value.trim(),
            backend: backend.value,
            model: model.value.trim(),
            description: description.value.trim(),
            permission_mode: mode.value,
        }),
    };
}
