# Review: Settings UI - profile switcher + informative panels

- TASK: 20260720-184149
- BRANCH: feature/settings-panels

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (fresh subagent; recreated the node_modules symlink,
  ran full `npm run ci`, removed the symlink after)

No BLOCKER/MAJOR/MINOR. `npm run ci` green (lint + tsc + 114 vitest + webpack
build). Reviewer verified: panel degradation (every row rendered, `value ?? "-"`,
5 cards even when empty), the context window-fill divide-by-zero guard
(`context_window > 0`), the profile switcher shown only when writable with the
active profile marked + its activate button disabled + no delete on it, confirm
on delete, single-authoritative re-render via dispatch->reload, XSS escaping on
all panel strings (profile names via textContent), and `renderSettings` purity
(extras optional, existing display tests unaffected).

- [x] R1.1 (NIT) settings-view.ts - noting the profile-switcher gate
  (`live = writable && actions`) is done correctly; no change needed.
  - Response: acknowledged; no change.

Open `manual:` DoD items (batched to the user at Finish): each panel shows real
data on the running server, and switching a profile is reflected in the page -
both were confirmed via the e2e serve during /work (page 200, profiles
list/create/activate, memory footprint of 27 real sessions, account live).
