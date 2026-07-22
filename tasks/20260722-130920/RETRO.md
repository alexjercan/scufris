# Retro: account auth_mode backend-aware (ChatGPT for codex, claude.ai for claude)

- TASK: 20260722-130920
- BRANCH: feature/account-auth-backend-aware (landed 8cc849a)
- REVIEW ROUNDS: 1 (out-of-context APPROVE, no findings)

See TASK.md for the pinned direction + spike findings and REVIEW.md for the
verification. Process notes only here.

## What went well

- I asked the user ONE targeted question about how they actually authenticate
  (browser/subscription vs api keys) before designing, and their answer ("ChatGPT
  and claude.ai like the browser login; I don't have api keys but keep them just
  in case") settled the whole model: a per-backend subscription default with an
  optional api_key. No guessing about a claude auth model the app cannot observe.
- The recon disambiguated the deferred question from the health task (extend the
  model vs return None): `agent_auth_mode` was already DISPLAY-ONLY at the report
  sites (its only behavioral use is the codex login gate), so extending the model
  was safe and honest - the reported value has the exact same status for both
  backends (a declared mode the CLI enforces).
- Mirrored the exact shape of the backend-aware-health fix (a pure
  `x_for_backend(settings, backend)` helper + dispatch every report site by the
  agent's own backend), so the change was mechanical, small, and the reviewer's
  "did you miss a site / dispatch the wrong backend" checks all passed first try.
- Enumerated ALL FOUR report sites up front (the lesson from the health task's
  single-site fix), so I did not leave the orchestrator info/config reporting the
  wrong auth while fixing only the per-agent panel.

## What went wrong

- Nothing substantive - the review found zero issues. The one friction was
  cosmetic: my `git commit -F` fallback logic (`... -F file 2>/dev/null || -m`)
  was defensive scaffolding I did not need; the `-F` path is the right default for
  a body with code punctuation (per the lesson from the styling task).

## What to improve next time

- Keep leaning on `git commit -F <file>` for any message with backticks/`->`/`$()`
  from the start (the `no-backticks-in-git-commit-m` lesson), rather than a
  try-`-F`-else-`-m` hedge.

## Action items

- [x] Nothing to adopt (APPROVE, no findings).
- Note (deferred, not this task): if the user ever wants scufris to PERFORM claude
  login (a `scufris login` claude flow + ANTHROPIC key handling), that is a
  separate feature - this task only made the REPORTED auth mode backend-correct.
