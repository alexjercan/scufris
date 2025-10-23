import re
from datetime import timedelta
from typing import List

from pydantic import BaseModel, Field


class Subtitle(BaseModel):
    index: str = Field(..., description="Subtitle index")
    start: timedelta = Field(..., description="Start time in seconds")
    end: timedelta = Field(..., description="End time in seconds")
    content: str = Field(..., description="Subtitle text")


TIME_RE = re.compile(r"(\d+):(\d+):(\d+),(\d+)")


def parse_timecode(tc: str) -> timedelta:
    """Parse a timecode string (e.g., "00:01:23,456") into a timedelta object."""

    m = TIME_RE.match(tc.strip())
    if not m:
        raise ValueError(f"Bad timecode: {tc}")
    hh, mm, ss, ms = map(int, m.groups())
    return timedelta(hours=hh, minutes=mm, seconds=ss, milliseconds=ms)


def srt_load(srt_path: str) -> List[Subtitle]:
    """Load subtitles from an SRT file."""

    with open(srt_path, encoding="utf-8") as f:
        content = f.read()

    return srt_loads(content)


def srt_loads(srt_content: str) -> List[Subtitle]:
    """Load subtitles from an SRT content string."""

    segments = []
    blocks = re.split(r"\n\s*\n", srt_content.strip())
    for block in blocks:
        lines = block.splitlines()
        if len(lines) >= 3:
            index = lines[0]
            times = lines[1]
            text = " ".join(lines[2:])
            if "-->" in times:
                start_tc, end_tc = times.split("-->")
                start = parse_timecode(start_tc)
                end = parse_timecode(end_tc)

                segments.append(
                    Subtitle(index=index, start=start, end=end, content=text)
                )
            else:
                continue

    return segments
