# Retro: clear 4 closed-unchecked tatr findings

## What went well

- Didn't blanket-tick: verified each unchecked step against evidence before
  ticking - RETRO corroboration for the manual smokes, and actual code for the
  impl claims (sessions.py logger at :64/:276/:314; agent-view runStreamingTurn
  and style.css .chat__thinking). The out-of-context reviewer re-verified every
  code claim against source and found no false ticks.
- Annotated each tick with a dated "Hygiene pass" note rather than silently
  flipping the box, and the manual-smoke annotations honestly disclose the tick
  is retroactive (they do not claim a re-run live smoke).

## What went wrong

- R1.1 (MINOR): task 235505's original step says "DEBUG counts for list/read"
  but only list_sessions logs a DEBUG count; the read paths log nothing. This is
  a pre-existing overclaim in that task's own step text. Left verbatim per the
  task-history immutability policy - the record shows what was written at close;
  my annotation cites only the real lines.

## What to improve next time

- closed-unchecked findings on shipped tasks are usually unticked manual smokes
  or impl steps, not undone work - but they must be checked individually against
  code/RETRO, never blanket-ticked. The honest tick + disclosure annotation is
  the pattern.

## Action items

- [x] 4 findings cleared; `tatr check` clean; landed e36ae08.
