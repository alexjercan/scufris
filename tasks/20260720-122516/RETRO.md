# Retro: diff-block rich preview (+ split 122516)

- DATE: 20260720
- VERDICT: shipped

## What went well

- Probing the codex image-input shape FIRST (via `codex app-server generate-ts` ->
  v2/UserInput.ts) answered the spike's open question decisively: both backends
  take a local image path (`-i` for exec, `{type:localImage, path}` for app_server).
  That let me make an evidence-based split instead of guessing - attachments is a
  de-risked-but-large vertical, so it became its own well-specified task, and this
  cycle shipped the self-contained diff-preview win.
- The diff renderer slotted cleanly into the existing markdown builder: extract the
  copy button, dispatch on `lang === "diff"`, build line rows with textContent. The
  XSS-free invariant carried over for free and is pinned by a hostile-diff test.

## What went wrong / friction

- Nothing. The only judgement call was the header-ordering bug class (`+++`/`---`
  must beat `+`/`-`), anticipated and tested.

## Lesson

- No new ledger entry. Reuses `build-dom-not-parse-html-for-untrusted-markdown`
  (textContent line rows) and `probe-runtime-on-target-host-early` (generate-ts
  probe before scoping the attachment feature).

## Follow-ups

- Image attachments: task 20260720-144530 (probed, specified, ready to flow).
- File-path chips (click a path to load its content): noted "later" in 122516; not
  yet a task - seed when wanted.
- Round-3 remaining: 122514 (den tools - waiting on the `today` CLI), 122518
  (projects sub-spike), 122519 (nixos reconcile), 134545 ("try it" runner).
