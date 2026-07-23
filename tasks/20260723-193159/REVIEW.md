# Review: reconcile agent_enabled default vs README docs drift

- TASK: 20260723-193159
- BRANCH: docs/agent-enabled-default
- DATE: 20260723
- REVIEWER: self (trivial-diff carve-out)
- VERDICT: APPROVE

## Verdict: APPROVE (trivial-diff self-review)

A 2-hunk, docs-only change to README.md (no code, no `.env.example` change - it
was already correct). Per the review skill's trivial-diff carve-out, no
out-of-context round was run.

### Checks
- Correctness: the new text matches `config.py:95` (`agent_enabled=True`) - agents
  are on by default; the flag disables (`=0`). Preserves the real caveat (inert
  until a backend is authenticated) that the old "off by default / provisioned by
  the operator" wording was gesturing at.
- No remaining "off by default" agent claim in README (`grep` clean); the
  redundant `export SCUFRIS_AGENT_ENABLED=1` removed from the quickstart.
- ASCII-clean; ruff/mypy/pytest green on the branch.
