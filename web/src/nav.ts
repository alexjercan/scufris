// Mark the current page's nav link active. Shared by every page.

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
}
