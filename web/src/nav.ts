// Mark the current page's nav link active, and wire the sign-out control.
// Shared by every page.

import { fetchJson, logout } from "./common";

interface AuthSession {
    authenticated: boolean;
    required: boolean;
}

export function initNav(): void {
    const path = window.location.pathname;
    const links = document.querySelectorAll<HTMLAnchorElement>(".nav__link");
    for (const link of links) {
        const href = link.getAttribute("href") ?? "";
        const active =
            href === "/"
                ? path === "/" || path === "/index.html"
                : href !== "" && path.startsWith(href);
        link.classList.toggle("is-active", active);
        if (active) link.setAttribute("aria-current", "page");
    }
    void initLogoutControl();
}

// Show "Sign out" only where there is a session to end: in loopback development
// authentication is off, and a control that logs you out of nothing is noise.
async function initLogoutControl(): Promise<void> {
    const button = document.getElementById(
        "nav-logout",
    ) as HTMLButtonElement | null;
    if (!button) return;
    try {
        const session = await fetchJson<AuthSession>("/api/auth/session");
        if (!session.required) return;
    } catch {
        return; // cannot tell; leave the control hidden rather than guess
    }
    button.hidden = false;
    button.onclick = () => {
        void logout();
    };
}
