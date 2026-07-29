// The sign-in form's behavior, kept free of import-time side effects so jsdom
// tests can import it without kicking off fetches or timers (lesson
// side-effect-free-module-for-jsdom-tests). `login.ts` is the thin entry.

export interface LoginDeps {
    // Injected so a test can drive the form without a network or a real
    // navigation (jsdom refuses to navigate).
    post: (password: string) => Promise<Response>;
    navigate: (url: string) => void;
}

// Where to land after a successful sign-in. The value arrives in the query
// string from the server's redirect, so it is untrusted: anything that is not a
// plain local path is discarded rather than repaired. This mirrors
// `auth.safe_next_path` on the server - an open redirect on a login page is a
// phishing primitive, and BOTH ends have to refuse it (the server sets the
// parameter, but anyone can hand the operator a link with their own).
export function safeNext(raw: string | null): string {
    if (!raw || !raw.startsWith("/")) return "/";
    if (raw.startsWith("//") || raw.startsWith("/\\")) return "/";
    if (raw.includes("\\")) return "/";
    return raw;
}

function messageFor(status: number, detail: string): string {
    if (status === 429) {
        return detail || "Too many attempts. Wait a bit and try again.";
    }
    if (status === 401) return "Wrong password.";
    return detail || `Sign-in failed (${String(status)}).`;
}

export function initLogin(deps: LoginDeps): void {
    const form = document.getElementById(
        "login-form",
    ) as HTMLFormElement | null;
    const input = document.getElementById(
        "password",
    ) as HTMLInputElement | null;
    const button = document.getElementById(
        "login-submit",
    ) as HTMLButtonElement | null;
    const error = document.getElementById("login-error");
    if (!form || !input || !button || !error) return;

    const showError = (text: string): void => {
        error.textContent = text;
        error.hidden = false;
    };

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        error.hidden = true;
        button.disabled = true;
        const password = input.value;
        void (async () => {
            try {
                const resp = await deps.post(password);
                if (resp.ok) {
                    const next = new URLSearchParams(
                        window.location.search,
                    ).get("next");
                    deps.navigate(safeNext(next));
                    return;
                }
                let detail = "";
                try {
                    const data = (await resp.json()) as { detail?: string };
                    detail = data.detail ?? "";
                } catch {
                    // non-JSON error body; the status carries the message
                }
                showError(messageFor(resp.status, detail));
            } catch {
                showError("Could not reach the server.");
            } finally {
                button.disabled = false;
                // Never leave the password sitting in the DOM after a failure.
                input.value = "";
                input.focus();
            }
        })();
    });
}
