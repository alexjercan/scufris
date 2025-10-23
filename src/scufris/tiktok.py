import argparse
import json
import logging
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import ffmpeg
import ollama
from faster_whisper import BatchedInferencePipeline, WhisperModel
from moviepy import (
    Clip,
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
)
from pydantic import BaseModel, Field, TypeAdapter
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.progress import track

from scufris.common import RichProgressBarLogger, get_logger, setup_logger, with_stats
from scufris.srt import Subtitle, srt_load

PROMPT_TEMPLATE = """You are an expert TikTok influencer and video‐analysis assistant.

Your task is to transform the provided transcript into a series of engaging, readable, and accessible subtitles suitable for a TikTok video. Follow these guidelines:

0. **Reliability**: Ensure that the subtitles contain the exact same text as the transcript, without adding or omitting any information.
1. **Conciseness & Clarity**: Each subtitle should convey the message in a concise manner, ensuring clarity and ease of reading (maximum 2 words per subtitle).
2. **Timing & Synchronization**: Ensure that subtitles are appropriately timed to match the speech, with a maximum duration of 2 seconds per subtitle.
3. **Engagement**: Use language that is engaging and relatable to a TikTok audience, incorporating elements like emojis or slang where appropriate.
4. **Accessibility**: Ensure that subtitles are readable by viewers with varying reading speeds and are synchronized with the audio.
5. **Formatting**: Present the subtitles in JSON format, adhering to the following structure:

```json
[
    {{
        "start": "00:00:01,000",
        "end": "00:00:05,000",
        "content": "Welcome to my TikTok!",
        "color": "#FFFFFF"
    }},
    ...
]
```

**EXAMPLES:**

*Example 1:*
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

*Improved JSON Output:*
```json
[
    {{
        "start": "00:00:00,000",
        "end": "00:00:01,000",
        "content": "Hey! 👋",
        "color": "#FFFFFF"
    }},
    {{
        "start": "00:00:01,000",
        "end": "00:00:03,000",
        "content": "What's up",
        "color": "#FFFFFF"
    }},
    {{
        "start": "00:00:03,000",
        "end": "00:00:04,000",
        "content": "everyone! 🙌",
        "color": "#FFFFFF"
    }},
]
```

*Example 2:*
```srt
1
00:00:00,000 --> 00:00:01,000
I

2
00:00:01,000 --> 00:00:02,000
have

3
00:00:02,000 --> 00:00:03,000
100

4
00:00:03,000 --> 00:00:04,000
tips for you!

```

*Improved JSON Output:*
```json
[
    {{
        "start": "00:00:00,000",
        "end": "00:00:02,000",
        "content": "I have",
        "color": "#FFFFFF"
    }},
    {{
        "start": "00:00:02,000",
        "end": "00:00:03,000",
        "content": "100",
        "color": "#FF00FF"
    }},
    {{
        "start": "00:00:03,000",
        "end": "00:00:04,000",
        "content": "tips for you! 💡",
        "color": "#FFFFFF"
    }},
]
```

**IMPORTANT**:
- Use at most 2 words per subtitle.
- Ensure each subtitle lasts no longer than 2 seconds.

*Now*, here is the transcript you need to improve:
```srt
{transcript}
```

Please produce the subtitles in JSON format to be added on the video.
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


class Config(BaseSettings):
    """Main configuration for the application."""

    model_config = SettingsConfigDict(env_prefix="SCUFRIS_TIKTOK_")

    projects_dir: Path = Field(
        default=Path(Path.cwd(), "projects"),
        description="Directory to store projects.",
    )


class WhisperConfig(BaseModel):
    """Configuration for the Whisper model."""

    model_config = SettingsConfigDict(env_prefix="SCUFRIS_TIKTOK_WHISPER_")

    model: str = Field(
        default="distil-large-v3",
        description="Faster Whisper model to use for transcription.",
    )
    device: str = Field(
        default="cpu",
        description="Device to run the Whisper model on.",
    )
    compute_type: str = Field(
        default="int8",
        description="Compute type for the Whisper model.",
    )
    beam_size: int = Field(
        default=5,
        description="Beam size for the Whisper model transcription.",
    )
    batch_size: int = Field(
        default=1,
        description="Batch size for the Whisper model transcription.",
    )
    vad_filter: bool = Field(
        default=True,
        description="Whether to apply VAD filtering during transcription.",
    )
    vad_min_silence_duration_ms: int = Field(
        default=2000,
        description="Minimum silence duration in ms for VAD filtering.",
    )


class OllamaConfig(BaseModel):
    """Configuration for Ollama."""

    model_config = SettingsConfigDict(env_prefix="SCUFRIS_TIKTOK_OLLAMA_")

    model: str = Field(
        default="qwen3",
        description="Ollama model to use for summarization.",
    )
    think: bool = Field(
        default=True,
        description="Whether to enable 'think' mode in Ollama.",
    )


@with_stats
def extract_audio(
    audio_path: Path, video_path: Path, logger: Optional[logging.Logger] = None
):
    """Extract audio from the given video file and save it as an audio file."""

    logger = logger or get_logger()
    logger.debug(f"Extracting audio from video: {video_path} to {audio_path}")

    (
        ffmpeg.input(video_path)
        .output(str(audio_path), ac=1, ar="16k")
        .overwrite_output()
        .run(quiet=True)
    )


def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp format."""

    millis = int((seconds - int(seconds)) * 1000)
    time_struct = time.gmtime(seconds)
    return time.strftime(f"%H:%M:%S,{millis:03d}", time_struct)


@with_stats
def create_transcript(
    transcript_path: Path,
    audio_path: Path,
    by_word: bool = False,
    logger: Optional[logging.Logger] = None,
    config: Optional[WhisperConfig] = None,
):
    """Create a transcript from the given audio file using Faster Whisper."""

    logger = logger or get_logger()
    config = config or WhisperConfig()
    logger.debug(
        f"Creating transcript for audio: {audio_path} to {transcript_path} using config: {config}"
    )

    model = WhisperModel(
        config.model,
        device=config.device,
        compute_type=config.compute_type,
    )
    args = dict(
        vad_filter=config.vad_filter,
        vad_parameters=dict(min_silence_duration_ms=config.vad_min_silence_duration_ms),
        beam_size=config.beam_size,
    )
    logger.debug(f"Whisper transcription arguments: {args}")

    if config.batch_size == 1:
        logger.debug("Using single inference pipeline for transcription.")
        segments, info = model.transcribe(
            str(audio_path), word_timestamps=by_word, **args
        )
    else:
        logger.debug(
            f"Using batched inference pipeline for transcription with batch size {config.batch_size}."
        )
        batched_model = BatchedInferencePipeline(model)
        segments, info = batched_model.transcribe(
            str(audio_path),
            word_timestamps=by_word,
            batch_size=config.batch_size,
            **args,
        )

    logger.debug(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )

    with open(transcript_path, "w", encoding="utf-8") as f:
        if by_word:
            for i, segment in track(
                enumerate(segments), description="Transcribing audio..."
            ):
                for j, word in enumerate(segment.words):
                    f.write(f"{i}_{j}\n")
                    f.write(
                        f"{format_timestamp(word.start)} --> {format_timestamp(word.end)}\n"
                    )
                    f.write(f"{word.word.strip()}\n\n")

        else:
            for i, segment in track(
                enumerate(segments), description="Transcribing audio..."
            ):
                f.write(f"{i}\n")
                f.write(
                    f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
                )
                f.write(f"{segment.text.strip()}\n\n")


@with_stats
def create_better_captions(
    better_path: Path,
    transcript_path: Path,
    logger: Optional[logging.Logger] = None,
    config: Optional[OllamaConfig] = None,
):
    logger = logger or get_logger()
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


class BetterSubtitle(BaseModel):
    start: timedelta = Field(..., description="Start time in seconds")
    end: timedelta = Field(..., description="End time in seconds")
    content: str = Field(..., description="Subtitle text")
    color: str = Field(alias="color", default="#FFFFFF", description="Subtitle color")


def clip_filename(subtitle: BetterSubtitle) -> str:
    return f"clip_{subtitle.start.total_seconds():.3f}_{subtitle.end.total_seconds():.3f}.mp4"


def create_caption_clips(
    run_path: Path,
    video_size: Tuple[int, int],
    word_timestamps: List[BetterSubtitle],
    font_size: int = 60,
    font: Optional[str] = None,
) -> List[Clip]:
    clips = []
    for w in track(word_timestamps, description="Creating caption clips..."):
        output_file = run_path / clip_filename(w)
        if not output_file.exists():
            duration = (w.end - w.start).total_seconds()
            txt = TextClip(
                font=None,
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
            .with_position(("center", font_size * 3.0))
            .with_start(w.start.total_seconds())
            .with_layer_index(1)
        )
        clips.append(clip)

    return clips


def subtitle_to_better(subtitle: Subtitle) -> BetterSubtitle:
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
def burn_in_captions(
    run_path: Path, input_video_path: Path, better_path: Path, output_path: Path
):
    video = VideoFileClip(str(input_video_path)).with_layer_index(0)
    video_size = video.size

    if better_path.suffix == ".json":
        with open(better_path, "r", encoding="utf-8") as f:
            better_captions = json.load(f)
            word_ts = TypeAdapter(List[BetterSubtitle]).validate_python(better_captions)
    elif better_path.suffix == ".srt":
        word_ts = [subtitle_to_better(s) for s in srt_load(str(better_path))]

    caption_clips = create_caption_clips(
        run_path,
        video_size,
        word_ts,
        font_size=int(video_size[0] * 0.1),
        font="iosevka",
    )

    print(f"Number of caption clips: {len(caption_clips)}")
    final = CompositeVideoClip([video, *caption_clips], size=video_size)
    final.write_videofile(
        output_path,
        fps=60,
        codec="libx264",
        audio_codec="aac",
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

    if args.better:
        better_path = Path(run_path, "better_captions.json")
        if not better_path.exists():
            create_better_captions(better_path, transcript_path)
            logger.info(f"Created better captions at: {better_path}")
        else:
            logger.info(f"Cached better captions found at: {better_path}")

        transcript_path = better_path

    output_video_path = args.output
    burn_in_captions(run_path, args.video_file, transcript_path, output_video_path)
    logger.info(f"Burned in captions to output video at: {output_video_path}")

    print(f"Run ID: {run_id} finished!")
    print(f"Output video saved at: {output_video_path}")
