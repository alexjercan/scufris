# Add the dashboard host approval queue and audit surface

- STATUS: OPEN
- PRIORITY: 44
- TAGS: feature, v0.2.0, host, frontend, ui

## Story

As the operator, I want a page in the dashboard that shows every host action
waiting on me - what it would run, what it would change, and how it can be undone -
so that approving a change to this machine is a decision I make from what is in
front of me rather than from an agent's paraphrase.

The routes already exist and are operator-only (`/api/host/actions`, `.../approve`,
`.../deny`, `.../cancel`, `.../revert`, `/api/host/audit`, and the per-action
event stream). This task is the surface over them, and it renders the confirmation
requirement the approval core computes rather than deciding for itself what
"stronger acknowledgement" means (`tasks/20260729-125040/DECISION.md`, sections 3
and 6).

## Steps

- [ ] Add the page: a `host` webpack entry, `web/src/host.html`, the FastAPI page
      route and its `historyApiFallback` rule, and a nav link. It must be
      protected by the deny-by-default middleware with no allowlist entry, and
      every call must go through `apiFetch` (the CSRF/401 seam).
- [ ] Render the pending queue: for each proposal its risk class, kind, EVERY
      command in order, the preview (kind, label, availability line, lines),
      who asked (operator vs agent plus its run), the expiry, and approve/deny
      controls. A multi-step action must never be summarised into one line.
- [ ] Make the risk class legible and the confirmation proportionate: R1 service
      control, R2 one-way cleanup and R3 config change must not look identical,
      the reversible ones show their undo sentence, and a one-way action's
      approve control is gated behind the acknowledgement token the core
      requires - the ordinary approve path must not be able to send it.
- [ ] Render the audit history with its rollback controls: an applied action that
      can be undone offers the revert, which PROPOSES the inverse and lands it
      back in the queue as its own proposal with its own preview and approval.
- [ ] Render the edges honestly: expired, drifted, already-decided, a cancelled
      or crashed apply, a 409 from a decision the other surface just made, a
      denial reason, empty and error states, and a hostd that is not configured
      (503) as "not configured" rather than as broken.
- [ ] Escape every host-supplied string. A systemd unit is named by a FILE and a
      preview quotes store paths and journal text, so this surface renders
      attacker-influenced text (stored XSS shipped in the cards built by
      20260729-125024 - review round 2).
- [ ] Stream an approved action's live output from
      `/api/host/actions/{id}/events` so a running switch shows progress rather
      than a spinner, and show a failed multi-step apply's partial-step detail.
- [ ] Cover desktop and phone widths; the approve/deny controls must be usable
      one-handed on a phone.

## Definition of Done

- The queue renders a proposal's commands, preview, requester, expiry and undo
  from a fixture with no fetch (test: `renderHostQueue` vitest suite).
- A one-way action cannot be approved from the ordinary control; the strong path
  is the only one that sends the acknowledgement
  (test: `test_one_way_approve_control_is_gated` vitest).
- Host-supplied text is escaped, unit names and preview lines included
  (test: `test_host_queue_escapes_host_text` vitest).
- The `/host` page and its API calls are protected by default - the page is in no
  public allowlist and the enumeration test still passes
  (test: `test_host_page_requires_a_session`).
- The audit view offers a rollback exactly where the record says one is possible,
  and the rollback appears as a new pending proposal
  (test: `test_revert_appears_as_a_new_proposal` vitest).
- cmd: `cd web && npm run ci`
- cmd: `python -m pytest`
- cmd: `nix flake check`
- manual: the queue is readable at phone width and the risk difference between a
  service restart and a system switch is obvious at a glance.

## Notes

- Epic: 20260729-124655. Depends on 20260729-125040 (the approval service and the
  confirmation requirement it exposes).
- Re-cut from 20260729-125040 - see `tasks/20260729-125040/DECISION.md` section 1.
- Existing views are PURE render functions driven by jsdom tests
  (`web/src/stats-view.ts`, `agents-view.ts`); follow that shape.
- Frontend ledger lessons that apply here: a render rewrite must sweep its CSS
  (`render-rewrite-orphans-its-css`), and a changed shared TS interface needs the
  webpack build, not just vitest (`type-change-fails-strict-tsc-not-vitest`).

## Flow State

- FLOW STEP: PLANNING
- PLAN STATUS: APPROVED
