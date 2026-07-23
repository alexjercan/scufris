# Retro: reconcile agent_enabled default vs README docs drift

- TASK: 20260723-193159
- DATE: 20260723
- OUTCOME: landed, trivial-diff self-review (no out-of-context round)

## What we set out to do

Fix the README drift where it claimed agents are "off by default" and told the
reader to `export SCUFRIS_AGENT_ENABLED=1`, while `config.py:95` is
`agent_enabled=True`. User chose "fix the docs, keep default=True" at the gate.

## What went well

- The sweep found that `.env.example` was ALREADY correct ("On by default ... Set
  to 0 to disable"), so only the two README spots were actually wrong - no
  scattershot editing. Checking the whole doc surface before editing kept the diff
  to exactly what drifted.
- Chose the honest framing over the literal one: agents are on by default BUT
  inert until a backend CLI is authenticated. That preserves the real caveat the
  old text was clumsily gesturing at ("provisioned by the operator") instead of
  just flipping "off" to "on".
- Dropped the redundant `export SCUFRIS_AGENT_ENABLED=1` from the quickstart
  rather than leaving a no-op that re-implies the flag is required.

## What went wrong / friction

- Nothing. Trivial docs task; the only judgement call was the gate fork (docs vs
  flipping the default), which the user resolved.

## Lessons

- None new. Reconfirms: reconcile a docs-vs-code drift by reading BOTH sides and
  the sibling doc surfaces (`.env.example` here was already right) before editing,
  so the fix lands only where the drift actually is.

## Deferred

- None. The gate's alternative (flip the default to False) was explicitly declined.
