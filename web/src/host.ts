// Entry for the host actions page: wire the view up and start polling. Kept
// separate from `host-view.ts` so that module has no import-time side effects and
// stays importable by the jsdom tests.

import "./style.css";
import { initNav } from "./nav";
import { startHost } from "./host-view";

initNav();
startHost();
