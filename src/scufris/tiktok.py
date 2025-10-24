import argparse
import json
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import ollama
from moviepy import (
    Clip,
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
)
from pydantic import BaseModel, Field, TypeAdapter
from rich.progress import track

from scufris.common import (
    Config,
    OllamaConfig,
    RichProgressBarLogger,
    Subtitle,
    create_transcript,
    extract_audio,
    get_logger,
    json_default,
    setup_logger,
    srt_load,
    with_stats,
)

PROMPT_TEMPLATE = """You are an expert TikTok subtitle editor and video-analysis assistant.

Your goal is to transform the provided transcript into a high-quality set of TikTok-style subtitles, formatted in JSON. The result must be *reliable, concise, and visually engaging* while preserving the transcript’s meaning exactly.

---

## 🎯 OBJECTIVE
Convert the transcript into short, readable subtitles that:
- Fit TikTok’s fast-paced visual style
- Never add or invent words
- May remove filler or redundant words (e.g., “uh,” “like,” “you know”)
- Keep subtitles easy to read on small screens

---

## 🧩 RULES

### 0. **Accuracy (NO HALLUCINATION)**
- Do **not** add, change, or rephrase words.
- You **may omit** useless filler words, repeated phrases, or noise tokens (e.g., “um,” “ah,” “like”).
- Never create sentences that didn’t exist in the transcript.

### 1. **Length**
- Each subtitle may contain **2–3 words maximum**.
- Break longer phrases into smaller chunks while keeping them meaningful.

### 2. **Timing**
- Each subtitle must have a duration of **≥ 0.5 seconds** and **≤ 2.0 seconds**.
- If input timestamps are broken or too short, normalize them (e.g., minimum 0.5s per segment).

### 3. **Engagement**
- Lightly enhance readability and emotional tone.
- You may add **occasional emojis** (max 1 per 3 subtitles).
- You may assign **color** per line to emphasize key moments or emotional tone.
  - Use HEX format (e.g. "#FFFFFF", "#FF00FF").
  - Keep color use subtle and varied.

### 4. **Readability**
- Subtitles should be readable to the average TikTok viewer at a glance.
- No line should exceed **3 words**.
- Keep punctuation minimal.

### 5. **Output Format**
Return subtitles **only** as a valid JSON array using the following structure:

```json
[
  {{
    "start": "00:00:01,000",
    "end": "00:00:02,000",
    "content": "What's up",
    "color": "#FFFFFF"
  }},
  ...
]
```

## ✅ EXAMPLES

### Input Transcript

```srt
1
00:00:00,000 --> 00:00:01,000
Hey!

2
00:00:01,000 --> 00:00:02,000
What's

3
00:00:02,000 --> 00:00:03,000
up

4
00:00:03,000 --> 00:00:04,000
everyone!
```

### Output

```json
[
  {{
    "start": "00:00:00,000",
    "end": "00:00:01,500",
    "content": "Hey! 👋",
    "color": "#FFFFFF"
  }},
  {{
    "start": "00:00:01,500",
    "end": "00:00:03,000",
    "content": "What's up",
    "color": "#FFFFFF"
  }},
  {{
    "start": "00:00:03,000",
    "end": "00:00:04,000",
    "content": "everyone!",
    "color": "#00FFAA"
  }}
]
```

## ⚙️ INSTRUCTIONS

1. Preserve all original text meaning.
2. Remove only filler or meaningless words.
3. Limit each subtitle to 2–3 words, with a minimum duration of 0.5 seconds.
4. Add color and occasional emojis tastefully.
5. Output only JSON, with no extra commentary or explanation.

Now, here is the transcript you need to process:

```srt
{transcript}
```
"""

JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "start": {"type": "string", "format": "HH:MM:SS,mmm"},
            "end": {"type": "string", "format": "HH:MM:SS,mmm"},
            "content": {"type": "string"},
            "color": {"type": "string"},
        },
        "required": ["start", "end", "content", "color"],
    },
}


class BetterSubtitle(BaseModel):
    """A subtitle with additional styling information."""

    start: timedelta = Field(..., description="Start time in seconds")
    end: timedelta = Field(..., description="End time in seconds")
    content: str = Field(..., description="Subtitle text")
    color: str = Field(alias="color", default="#FFFFFF", description="Subtitle color")


def __subtitle_to_better(subtitle: Subtitle) -> BetterSubtitle:
    """Convert a Subtitle to a BetterSubtitle with default styling."""

    color = "#FFFFFF"
    if subtitle.content.strip().isdigit():
        color = "#FFFF00"

    return BetterSubtitle(
        start=subtitle.start,
        end=subtitle.end,
        content=subtitle.content,
        color=color,
    )


@with_stats
def create_better_captions(
    better_path: Path,
    transcript_path: Path,
    use_llm: bool = True,
    logger: Optional[logging.Logger] = None,
    config: Optional[OllamaConfig] = None,
):
    logger = logger or get_logger()

    if not use_llm:
        logger.debug("Skipping LLM caption improvement; using simple conversion.")

        word_ts = [__subtitle_to_better(s) for s in srt_load(str(transcript_path))]
        with open(better_path, "w", encoding="utf-8") as f:
            json.dump(
                [w.dict(by_alias=True) for w in word_ts],
                f,
                ensure_ascii=False,
                indent=4,
                default=json_default,
            )

        return

    config = config or OllamaConfig()
    logger.debug(
        f"Creating better content for transcript: {transcript_path} to {better_path} using config: {config}"
    )

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    prompt = PROMPT_TEMPLATE.format(transcript=transcript_text)
    prompt_path = better_path.with_suffix(".prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    response = ollama.generate(
        model=config.model,
        prompt=prompt,
        think=config.think,
        format=JSON_SCHEMA,
    )

    with open(better_path, "w", encoding="utf-8") as f:
        f.write(response.response)


def __clip_filename(subtitle: BetterSubtitle) -> str:
    return f"clip_{subtitle.start.total_seconds():.3f}_{subtitle.end.total_seconds():.3f}.mp4"


def create_caption_clips(
    run_path: Path,
    video_size: Tuple[int, int],
    word_timestamps: List[BetterSubtitle],
) -> List[Clip]:
    clips = []
    font_size = int(video_size[0] * 0.1)

    for w in track(word_timestamps, description="Creating caption clips..."):
        output_file = run_path / __clip_filename(w)
        if not output_file.exists():
            duration = (w.end - w.start).total_seconds()
            assert duration > 0, f"Invalid duration for subtitle: {w}, please edit manually {run_path / 'better_captions.json'}"

            txt = TextClip(
                font="./fonts/Iosevka-Regular.ttf",
                font_size=font_size,
                text=w.content,
                color=w.color,
                duration=duration,
                size=video_size,
            )

            txt.write_videofile(str(output_file), codec="png", fps=60, logger=None)

            clip = txt

        clip = (
            VideoFileClip(str(output_file), has_mask=True)
            .with_position(("center", font_size * 4.0))
            .with_start(w.start.total_seconds())
            .with_layer_index(1)
        )
        clips.append(clip)

    return clips


@with_stats
def burn_in_captions(
    run_path: Path, input_video_path: Path, better_path: Path, output_path: Path
):
    video = VideoFileClip(str(input_video_path)).with_layer_index(0)
    video_size = video.size

    with open(better_path, "r", encoding="utf-8") as f:
        better_captions = json.load(f)
        word_ts = TypeAdapter(List[BetterSubtitle]).validate_python(better_captions)

    caption_clips = create_caption_clips(
        run_path,
        video_size,
        word_ts,
    )

    final = CompositeVideoClip([video, *caption_clips], size=video_size)
    final.write_videofile(
        output_path,
        fps=60,
        codec="libx264",
        bitrate="5000k",
        audio_codec="aac",
        audio_bitrate="192k",
        threads=4,
        logger=RichProgressBarLogger(),
    )


class Arguments(BaseModel):
    video_file: Path = Field(
        alias="video_file", description="Path to the input video file."
    )
    output: Path = Field(
        alias="output",
        default=Path("output.mp4"),
        description="Path to the output merged video file.",
    )
    better: bool = Field(
        alias="better",
        default=False,
        description="Whether to create better captions using Ollama.",
    )


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(
        description="Highlights video generator using Ollama and Faster Whisper."
    )
    parser.add_argument("video_file", type=str, help="Path to the input video file.")
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Path to the output merged video file.",
    )
    parser.add_argument(
        "--better",
        action="store_true",
        help="Whether to create better captions using Ollama.",
    )

    args = parser.parse_args()

    return Arguments(video_file=args.video_file, output=args.output, better=args.better)


def tiktok() -> None:
    config = Config()
    args = parse_arguments()

    run_id = str(args.video_file.stem)
    logger = setup_logger(run_id)

    logger.info("Starting tiktok process with run ID: %s", run_id)
    logger.debug(f"Loaded configuration: {config}")
    logger.debug(f"Parsed arguments: {args}")

    run_path = Path(config.projects_dir, run_id)
    os.makedirs(run_path, exist_ok=True)
    logger.debug(f"Created run directory at: {run_path}")

    audio_path = Path(run_path, "extracted_audio.wav")
    if not audio_path.exists():
        extract_audio(audio_path, args.video_file)
        logger.info(f"Extracted audio to: {audio_path}")
    else:
        logger.info(f"Cached audio found at: {audio_path}")

    transcript_path = Path(run_path, "transcript.srt")
    if not transcript_path.exists():
        create_transcript(transcript_path, audio_path, by_word=True)
        logger.info(f"Created transcript at: {transcript_path}")
    else:
        logger.info(f"Cached transcript found at: {transcript_path}")

    better_path = Path(run_path, f"better_captions_{int(args.better)}.json")
    if not better_path.exists():
        create_better_captions(better_path, transcript_path, use_llm=args.better)
        logger.info(f"Created better captions at: {better_path}")
    else:
        logger.info(f"Cached better captions found at: {better_path}")

    output_video_path = args.output
    burn_in_captions(run_path, args.video_file, better_path, output_video_path)
    logger.info(f"Burned in captions to output video at: {output_video_path}")

    print(f"Run ID: {run_id} finished!")
    print(f"Output video saved at: {output_video_path}")
