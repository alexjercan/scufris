"""Changelog and version plumbing shared by the release scripts and the tests.

`pyproject.toml` holds the version. `CHANGELOG.md` (Keep a Changelog) holds
what each version contains. A release is only coherent when the tag, the
project version, and the changelog's top released section all name the SAME
version - so that agreement is computed in one place here, and everything else
(the shell wrappers in this directory, the release workflow, the tests) calls
into it rather than re-implementing a parser.

Run as a CLI:

    python -m scripts.release_tools version
    python -m scripts.release_tools notes 0.1.0
    python -m scripts.release_tools cut 0.1.0 [--date YYYY-MM-DD]
    python -m scripts.release_tools check 0.1.0

Every subcommand exits non-zero with a message on stderr when the repository
is not in the state it claims to be. Nothing here prints a reassuring message
it has not verified.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

#: Repo root, relative to this file (scripts/release_tools.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent

UNRELEASED = "Unreleased"

#: `## [1.2.3] - 2026-07-29` or `## [Unreleased]`. The date is optional so a
#: half-written section is parsed and then REJECTED with a clear message,
#: rather than silently not matching and being reported as "missing".
#
# `[^\S\n]` (horizontal whitespace) rather than `\s` throughout: `\s` matches
# newlines, so with re.MULTILINE the optional date group happily reached across
# the blank line and captured the section's first list item as the date - which
# made an undated section look dated and an empty one look full.
_SECTION_RE = re.compile(
    r"^##[^\S\n]+\[(?P<version>[^\]]+)\][^\S\n]*"
    r"(?:-[^\S\n]*(?P<date>\S+))?[^\S\n]*$",
    re.MULTILINE,
)

#: A link-reference line at the foot of the changelog: `[1.2.3]: https://...`.
_LINK_RE = re.compile(r"^\[(?P<version>[^\]]+)\]:\s*(?P<url>\S+)\s*$", re.MULTILINE)

REPO_URL = "https://github.com/alexjercan/scufris"


class ReleaseError(Exception):
    """A release invariant does not hold. The message is for the operator."""


@dataclass(frozen=True)
class Section:
    """One `## [version] - date` block of the changelog."""

    version: str
    date: str | None
    body: str

    @property
    def is_unreleased(self) -> bool:
        return self.version == UNRELEASED

    @property
    def is_empty(self) -> bool:
        """No prose at all, or only empty `### Group` headings.

        A section holding nothing but `### Added` is empty for release-notes
        purposes: publishing it would produce a release page that says the
        version added nothing.
        """
        for line in self.body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("###"):
                continue
            return False
        return True


def _normalize(version: str) -> str:
    """`v1.2.3` and `1.2.3` are the same version. Tags carry the `v`."""
    return version[1:] if version.startswith("v") else version


#: PEP 440's pre-release and development spellings, after the release segment:
#: `1.0.0a1`, `1.0.0b2`, `1.0.0rc1`, `1.0.0.dev4`, and the separator-tolerant
#: forms (`1.0.0-rc.1`) that a git tag is likely to carry. Note what is NOT
#: here: `.postN` is a POST-release - later than the release, not earlier - and
#: a naive "anything beyond MAJOR.MINOR.PATCH is a pre-release" rule wrongly
#: marks it as one.
_PRERELEASE_RE = re.compile(
    r"^\d+(?:\.\d+)*(?:[-_.]?(?:a|b|c|rc|alpha|beta|pre|preview|dev)[-_.]?\d*)",
    re.IGNORECASE,
)


def is_prerelease(version: str) -> bool:
    """Whether this version is a pre-release, so the page can be marked as one.

    A release page that presents a candidate as final is a worse error than one
    that is over-cautious, but marking a `.post1` as a candidate is also wrong,
    so this follows PEP 440 rather than "has a suffix".
    """
    return bool(_PRERELEASE_RE.match(_normalize(version)))


def _normalize_newlines(text: str) -> str:
    """CRLF in, LF out.

    A changelog edited on Windows (or fetched through something that rewrites
    line endings) would otherwise put a stray `\\r` at the end of every line of
    the extracted release notes, and a rewritten file would end up with mixed
    endings.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _fenced_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges covered by fenced code blocks (``` or ~~~).

    A changelog entry may quote markdown, and a `## [1.0.0]` line inside a code
    fence is an example, not a section. Without this the parser invented
    sections and truncated the real ones at the fence.
    """
    spans: list[tuple[int, int]] = []
    open_at: int | None = None
    open_line = 0
    fence = ""
    position = 0
    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in ("```", "~~~"):
            if open_at is None:
                open_at = position
                open_line = number
                fence = marker
            elif marker == fence:
                spans.append((open_at, position + len(line)))
                open_at = None
        position += len(line)
    if open_at is not None:
        # An unterminated fence makes the whole tail of the document invisible:
        # every section below it would be swallowed, and a cut would then act on
        # a truncated picture - regenerating the wrong links and folding the
        # previous release's notes into the new section, all with exit 0. A
        # malformed changelog must stop the release, not be quietly reshaped.
        raise ReleaseError(
            f"CHANGELOG.md has an unterminated code fence opened at line {open_line}"
        )
    return spans


def _section_matches(text: str) -> list[re.Match[str]]:
    """Every real section heading - fenced examples excluded."""
    spans = _fenced_spans(text)
    return [
        match
        for match in _SECTION_RE.finditer(text)
        if not any(start <= match.start() < end for start, end in spans)
    ]


def parse_changelog(text: str) -> list[Section]:
    """Split a Keep a Changelog document into its `## [version]` sections.

    Everything before the first section (title, format blurb) and the trailing
    link-reference block are not sections and are not returned; use
    `split_document` when you need to rebuild the file.
    """
    text = _normalize_newlines(text)
    matches = _section_matches(text)
    sections: list[Section] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        # The link-reference block at the foot belongs to the document, not to
        # the last section.
        body = _LINK_RE.sub("", body)
        sections.append(
            Section(
                version=match.group("version").strip(),
                date=(match.group("date") or None),
                body=body.strip("\n"),
            )
        )
    return sections


def released_sections(sections: list[Section]) -> list[Section]:
    return [s for s in sections if not s.is_unreleased]


def find_section(sections: list[Section], version: str) -> Section | None:
    wanted = _normalize(version)
    for section in sections:
        if _normalize(section.version) == wanted:
            return section
    return None


def release_notes(text: str, version: str) -> str:
    """The body of `version`'s section, exactly - what the release page shows.

    Raises `ReleaseError` when the version has no section (a release must not
    silently publish empty notes) or when the section is empty.
    """
    wanted = _normalize(version)
    section = find_section(parse_changelog(text), wanted)
    if section is None:
        raise ReleaseError(
            f"CHANGELOG.md has no section for {wanted}. "
            f"Cut it first: scripts/cut-changelog.sh {wanted}"
        )
    if section.is_unreleased:
        raise ReleaseError(
            f"{wanted} resolves to the [Unreleased] section; a release needs a "
            "cut section with a date."
        )
    if section.date is None:
        raise ReleaseError(f"CHANGELOG.md section [{wanted}] has no release date.")
    if section.is_empty:
        raise ReleaseError(
            f"CHANGELOG.md section [{wanted}] is empty; there is nothing to release."
        )
    return section.body


def project_version(pyproject_text: str) -> str:
    data = tomllib.loads(pyproject_text)
    try:
        return str(data["project"]["version"])
    except KeyError as exc:  # pragma: no cover - malformed pyproject
        raise ReleaseError("pyproject.toml has no [project].version") from exc


def check_agreement(
    *,
    pyproject_text: str,
    changelog_text: str,
    tag: str | None = None,
) -> str:
    """Assert the version sources agree, and return the agreed version.

    The sources are `pyproject.toml`'s version, the changelog's TOP RELEASED
    section, and (when releasing) the tag. Disagreement raises, naming every
    value seen - a message that says only "mismatch" costs another round trip.
    """
    project = project_version(pyproject_text)
    released = released_sections(parse_changelog(changelog_text))
    if not released:
        raise ReleaseError(
            "CHANGELOG.md has no released section - only [Unreleased]. "
            f"Cut it: scripts/cut-changelog.sh {project}"
        )
    top = released[0]
    top_version = _normalize(top.version)
    if top_version != project:
        raise ReleaseError(
            "version sources disagree: pyproject.toml says "
            f"{project}, CHANGELOG.md's top released section says {top_version}"
        )
    if top.date is None:
        raise ReleaseError(f"CHANGELOG.md section [{top.version}] has no date")
    if tag is not None and _normalize(tag) != project:
        raise ReleaseError(
            f"version sources disagree: tag says {_normalize(tag)}, "
            f"pyproject.toml and CHANGELOG.md say {project}"
        )
    return project


def split_document(text: str) -> tuple[str, str, str]:
    """Return (preamble, sections_block, links_block).

    The links block is the trailing run of `[x]: url` lines; the preamble is
    everything before the first `## [` heading.
    """
    text = _normalize_newlines(text)
    matches = _section_matches(text)
    if not matches:
        raise ReleaseError("CHANGELOG.md has no '## [version]' sections")
    first = matches[0]
    preamble = text[: first.start()]
    rest = text[first.start() :]

    lines = rest.splitlines()
    tail_start = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if _LINK_RE.fullmatch(stripped):
            tail_start = i
            continue
        break
    sections_block = "\n".join(lines[:tail_start]).rstrip("\n")
    links_block = "\n".join(line for line in lines[tail_start:] if line.strip()).rstrip(
        "\n"
    )
    return preamble, sections_block, links_block


def _link_lines(versions: list[str], links_block: str) -> str:
    """Rebuild the link-reference block for `versions`, newest first.

    `[Unreleased]` compares the newest released tag against HEAD; each released
    version links to its tag comparison against the previous one, and the
    oldest links to the tag itself (there is nothing before it to compare to).
    Any pre-existing line for a version we do not manage is preserved.
    """
    # Compare NORMALIZED versions: a hand-written `[v1.0.0]: ...` line names the
    # same version as the generated `[1.0.0]: ...` one, and preserving it would
    # leave the file with two link references for one version.
    managed = {UNRELEASED, *versions}
    preserved = [
        line
        for line in links_block.splitlines()
        if (m := _LINK_RE.fullmatch(line.strip()))
        and m.group("version") not in managed
        and _normalize(m.group("version")) not in managed
    ]
    out: list[str] = []
    if versions:
        out.append(f"[{UNRELEASED}]: {REPO_URL}/compare/v{versions[0]}...HEAD")
    for i, version in enumerate(versions):
        if i + 1 < len(versions):
            previous = versions[i + 1]
            out.append(f"[{version}]: {REPO_URL}/compare/v{previous}...v{version}")
        else:
            out.append(f"[{version}]: {REPO_URL}/releases/tag/v{version}")
    return "\n".join(out + preserved)


def cut_changelog(text: str, version: str, date: str, *, redate: bool = False) -> str:
    """Turn `[Unreleased]` into `[version] - date` and open a fresh Unreleased.

    Idempotent: if `version` is already a cut section, the text is returned
    unchanged. That is what makes it safe for the release pipeline to run the
    same script the operator already ran by hand, on any later day.

    `redate=True` is the deliberate escape hatch for a release that slips: it
    corrects an already-cut section's date. It is OFF by default, and `main()`
    only turns it on when the operator passed `--date` explicitly. Making
    re-dating the default would mean a dateless re-run silently moved a
    published version's release date - which is the opposite of idempotent.

    Raises `ReleaseError` when there is no Unreleased section to cut, or when
    it is empty (a version whose notes say nothing must not be published).
    """
    text = _normalize_newlines(text)
    wanted = _normalize(version)
    sections = parse_changelog(text)

    existing = find_section(sections, wanted)
    if existing is not None and not existing.is_unreleased:
        if not redate or existing.date == date:
            return text  # already cut - no-op
        return _redate(text, existing.version, date)

    unreleased = next((s for s in sections if s.is_unreleased), None)
    if unreleased is None:
        raise ReleaseError("CHANGELOG.md has no [Unreleased] section to cut")
    if unreleased.is_empty:
        raise ReleaseError(
            f"[Unreleased] is empty; there is nothing to release as {wanted}"
        )

    preamble, sections_block, links_block = split_document(text)
    # Splice at the heading the PARSER found, not at a literal `## [Unreleased]`
    # string: the two disagree whenever the heading has unusual spacing, and a
    # str.replace that silently matched nothing used to produce a changelog with
    # no released section and no links, while reporting success.
    headings = _section_matches(sections_block)
    heading = headings[0] if headings else None
    if heading is None or heading.group("version").strip() != UNRELEASED:
        raise ReleaseError(
            "CHANGELOG.md's first section is not [Unreleased]; refusing to cut"
        )
    cut = (
        sections_block[: heading.end()]
        + f"\n\n## [{wanted}] - {date}"
        + sections_block[heading.end() :]
    )
    versions = [
        _normalize(s.version)
        for s in released_sections(parse_changelog(preamble + cut))
    ]
    links = _link_lines(versions, links_block)
    return f"{preamble}{cut}\n\n{links}\n"


def _redate(text: str, section_version: str, date: str) -> str:
    """Rewrite one already-cut section's date, leaving everything else alone."""
    for match in _section_matches(text):
        if match.group("version").strip() != section_version:
            continue
        return (
            text[: match.start()]
            + f"## [{section_version}] - {date}"
            + text[match.end() :]
        )
    # Unreachable: the caller located this section through the same parser. If
    # it ever fires, something is wrong enough that silently returning the text
    # unchanged (and reporting a successful re-date) would be the worst answer.
    raise ReleaseError(f"CHANGELOG.md section [{section_version}] vanished mid-edit")


def today() -> str:
    """Today as `YYYY-MM-DD`.

    A named seam rather than an inline `date.today()` so a test can pin what
    "today" is and assert that a dateless re-run does not move a release date.
    """
    return _datetime.date.today().isoformat()


def valid_date(value: str) -> str:
    """Accept only an ISO `YYYY-MM-DD` date.

    `--date` is the single gesture that authorizes overwriting an already-cut
    version's release date, so a typo must not be able to both re-date the
    section AND write nonsense into it in one step.
    """
    try:
        _datetime.date.fromisoformat(value)
    except ValueError as exc:
        raise ReleaseError(
            f"--date must be an ISO date (YYYY-MM-DD), not {value!r}"
        ) from exc
    return value


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ReleaseError(f"missing file: {path}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release_tools")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="print the version from pyproject.toml")
    notes = sub.add_parser("notes", help="print a version's changelog body")
    notes.add_argument("version")
    cut = sub.add_parser("cut", help="cut [Unreleased] into a released section")
    cut.add_argument("version")
    cut.add_argument("--date", default=None, help="release date (default: today)")
    cut.add_argument(
        "--check",
        action="store_true",
        help="verify the cut is already done instead of writing",
    )
    check = sub.add_parser("check", help="verify the version sources agree")
    check.add_argument("version", nargs="?", default=None, help="tag to check too")
    classify = sub.add_parser(
        "prerelease", help="print true/false: is this version a pre-release"
    )
    classify.add_argument("version")

    args = parser.parse_args(argv)
    root: Path = args.root
    pyproject = root / "pyproject.toml"
    changelog = root / "CHANGELOG.md"

    try:
        if args.command == "version":
            print(project_version(_read(pyproject)))
            return 0

        if args.command == "notes":
            print(release_notes(_read(changelog), args.version))
            return 0

        if args.command == "cut":
            text = _read(changelog)
            # An explicit --date is also the operator saying "I mean this date",
            # which is the only thing that authorizes re-dating an already-cut
            # section. Without it the command is idempotent by construction: a
            # dateless re-run on any later day leaves a cut file byte-identical
            # instead of quietly moving a published version's release date.
            date = valid_date(args.date) if args.date is not None else today()
            redate = args.date is not None
            # --check asks a question about the CURRENT file, so it must not go
            # through cut_changelog: a version that was never cut would report
            # whatever stopped the cut ("[Unreleased] is empty") instead of the
            # true answer ("not cut for X").
            if args.check:
                section = find_section(parse_changelog(text), _normalize(args.version))
                if section is None or section.is_unreleased:
                    raise ReleaseError(
                        f"CHANGELOG.md is not cut for {_normalize(args.version)}"
                    )
                print(f"CHANGELOG.md is cut for {_normalize(args.version)}")
                return 0
            updated = cut_changelog(text, args.version, date, redate=redate)
            if updated == text:
                print(f"CHANGELOG.md already cut for {_normalize(args.version)}")
                return 0
            changelog.write_text(updated, encoding="utf-8")
            print(f"cut CHANGELOG.md for {_normalize(args.version)} ({date})")
            return 0

        if args.command == "prerelease":
            print("true" if is_prerelease(args.version) else "false")
            return 0

        if args.command == "check":
            version = check_agreement(
                pyproject_text=_read(pyproject),
                changelog_text=_read(changelog),
                tag=args.version,
            )
            print(f"version sources agree on {version}")
            return 0
    except ReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0  # pragma: no cover - argparse requires a subcommand


if __name__ == "__main__":
    raise SystemExit(main())
