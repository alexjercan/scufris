# Review: Split the host, hostd, and auth modules under the size cap

- TASK: 20260731-171430
- BRANCH: refactor/split-host-hostd-auth
- BASE: master @ 089ea39

## Round 1

- VERDICT: APPROVE
- REVIEWER: primary, in session. The skill's default is an out-of-context
  reviewer; this session carries a standing instruction not to spawn subagents
  unless the user asks. Recorded as an exception rather than worked around. To
  compensate, every load-bearing claim was re-derived MECHANICALLY rather than
  by re-reading: see "Independent derivation" below, which is stronger evidence
  for a pure-move refactor than a second reader would have been. A fresh-context
  round is available on request.

### Independent derivation

Two checks were written against the diff rather than trusting the close-out.

**1. No public name silently left a facade.** For each of the four base modules,
parsed `git show master:<path>` with `ast` for module-level public names, then
compared with `dir()` on the imported package:

| Module | Base public names | Missing after the split |
|-|-|-|
| `scufris.hostd.actions` | 28 | none |
| `scufris.auth` | 26 | `logger` |
| `scufris.mcp_host_tools` | 24 | `logger` |
| `scufris.hostconfig` | 28 | `logger`, `MAX_CHANGES`, `MAX_LOG_TAIL`, `GIT_TIMEOUT`, `EVAL_TIMEOUT`, `Propose` |

Each narrowing was then checked for consumers across `scufris/`, `tests/` and
`examples/`: none, except `Propose`, whose single use is inside
`hostconfig/changes.py` itself. No logger name is configured by string anywhere
(`scufris/logsetup.py` has no `getLogger("...")` literals), so the per-submodule
logger names are inert. The narrowing is correct; see R1.2 on recording it.

**2. It is a pure move.** Normalised every base module and every submodule to a
multiset of stripped, non-blank, non-comment, non-import code lines and
differenced them. Across all four packages the ONLY lines lost are docstring
prose (deliberately reworded `module` -> `package`), the swept lore citations,
the two dead `logger` assignments, and the two `_validate_provenance` ->
`validate_provenance` rename lines. Excluding the facades, the only added
executable lines are `logger = logging.getLogger(__name__)` in a second auth
submodule, import continuations, and the two rename lines. **Zero logic lines
were added, removed, or reordered in any of the four modules** - which is the
claim the whole task rests on, and it is now measured rather than asserted.

That covers the security lane directly: `OPERATOR_ONLY_PATTERN`, the scrypt
parameters, `token_matches`'s `surrogatepass` totality, the 0600 atomic session
flush and the throttle's global ceiling are all provably verbatim, and
`scufris/app.py` - where the one deny-by-default middleware lives - is not in
the diff at all.

### Checks rerun

| Check | Result |
|-|-|
| `python -m pytest -o addopts=""` | 896 passed, matching the recorded baseline |
| targeted 7-file host/auth run | 220 passed |
| `python scripts/check_file_size.py` | green; 0 hits for the four owned entries |
| `git rebase master --exec 'python scripts/check_file_size.py'` | executed and passed at all 4 commits |
| `nix flake check` | ruff, mypy, pytest, filesize green. `records` reports the known tatr-0.1.0 false positive ("IN_PROGRESS task lacks PLAN STATUS: APPROVED" while the tracked file carries `PLAN STATUS: APPROVED`); it passed earlier in the round only because the committed record was still at PLANNING. Not chased, pin not bumped |
| `nix build .#scufris-hostd-vm-test` | built |
| `git diff master -- tests/ examples/` | 2 files, 7 lines, every one a monkeypatch target string; no import line changed |
| `rg '^PUBLIC_PATHS\|^PUBLIC_STATIC_PATHS' scufris/` | 2 hits, both `scufris/auth/policy.py` |
| `rg -n 'Depends' scufris/` | no hits |
| lore sweep over the four trees | no hits |
| `git diff master -- scripts/check_file_size.py` | exactly the 4 owned entries removed; `scufris/app.py`, 7 `tests/` and 5 `web/src/` entries untouched |

The two sabotage checks in the close-out were re-run and reproduce: reverting
one `_inspector` target gives 4 failed / 21 passed; reverting the
`scufris.auth.store.time.time` targets gives 2 failed. The close-out's
correction of the plan's prediction (loud, not silent) is accurate.

The `ruff format --check tests/` claim was re-derived the way the epic's rule
demands, since the close-out notes an earlier attempt at it went wrong:
`git show master:tests/test_host_mcp_server.py` into a scratch path and checked
there reports "Would reformat" on the BASE file. Inherited, confirmed
independently. The gate runs `ruff check .`, not the formatter, so it is out of
scope either way.

### Findings

**R1.1 NIT - `scufris/host/README.md:35`.** The line now reads "Registered by
`../mcp_host_tools/inspection.py`". The tools are DEFINED there; they are
registered by `register()` in `../mcp_host_tools/__init__.py`, which is the
whole point of the audience split the package docstring describes. The pre-split
text named the module that did both, so the split made this less accurate rather
than more. Change to name the definition site and the registrar separately, or
revert to naming the package (`../mcp_host_tools/`), as line 108 does. Rewrap:
the edit pushed the line past the surrounding width.

**R1.2 NIT - `tasks/20260731-171430/TASK.md`, close-out.** The close-out records
the two dropped `logger` globals but not the other five names that stopped being
reachable as `scufris.hostconfig.X`: `MAX_CHANGES`, `MAX_LOG_TAIL`,
`GIT_TIMEOUT`, `EVAL_TIMEOUT` and the `Propose` type alias. The narrowing is
CORRECT - verified above that nothing outside the package consumes any of them,
and the epic's rule is to export only names with a real consumer - but it is an
undeclared public-surface change and belongs next to the logger note, so a later
reader does not have to rediscover it. Fold it into the "logger dropped"
paragraph.

**R1.3 NIT - `scufris/hostd/actions/taxonomy.py:14` and the close-out's lore
count.** `RiskClass`'s docstring lost "the spike's" ("Which class of the spike's
taxonomy an action belongs to" -> "Which class of the taxonomy..."). That is a
correct sweep under the comment policy - a spike reference as the only
justification - but it is a TENTH site, and the close-out says "Nine lore sites
swept" and enumerates nine. Correct the count and add the site, so the record
matches what landed.

### Responses

All three NITs fixed on this branch before landing.

- R1.1 fixed: `scufris/host/README.md:35` now reads "Defined in
  `../mcp_host_tools/inspection.py` and registered by `mcp_host_tools.register`",
  rewrapped.
- R1.2 fixed: the close-out records the five narrowed `hostconfig` names and why.
- R1.3 fixed: the count reads "Ten lore sites" and names the `RiskClass` site.

### Pending manual items

None from this task. The epic's two manual acceptance items (20260731-171411)
stay pending until its remaining children land.
