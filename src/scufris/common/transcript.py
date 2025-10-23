import logging
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

from faster_whisper import BatchedInferencePipeline, WhisperModel
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict
from rich.progress import track

from .logging import get_logger, with_stats

TIME_RE = re.compile(r"(\d+):(\d+):(\d+),(\d+)")


class WhisperConfig(BaseModel):
    """Configuration for the Whisper model."""

    model_config = SettingsConfigDict(env_prefix="SCUFRIS_WHISPER_")

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


def __format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp format."""

    millis = int((seconds - int(seconds)) * 1000)
    time_struct = time.gmtime(seconds)
    return time.strftime(f"%H:%M:%S,{millis:03d}", time_struct)


def format_timedelta(td: timedelta) -> str:
    """Format timedelta to SRT timestamp format."""

    total_seconds = int(td.total_seconds())
    return __format_timestamp(total_seconds + td.microseconds / 1_000_000)


def parse_timecode(tc: str) -> timedelta:
    """Parse a timecode string (e.g., "00:01:23,456") into a timedelta object."""

    m = TIME_RE.match(tc.strip())
    if not m:
        raise ValueError(f"Bad timecode: {tc}")
    hh, mm, ss, ms = map(int, m.groups())
    return timedelta(hours=hh, minutes=mm, seconds=ss, milliseconds=ms)


@with_stats
def create_transcript(
    transcript_path: Path,
    audio_path: Path,
    by_word: bool = False,
    logger: Optional[logging.Logger] = None,
    config: Optional[WhisperConfig] = None,
):
    """Create a transcript from the given audio file using Faster Whisper.

    Args:
        transcript_path (Path): Path to save the transcript file.
        audio_path (Path): Path to the input audio file.
        by_word (bool): Whether to create a word-level transcript. Defaults to False.
        logger (Optional[logging.Logger]): Logger instance for logging. If None, a default logger is used.
        config (Optional[WhisperConfig]): Configuration for the Whisper model. If None, default config is used.

    """

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
                        f"{__format_timestamp(word.start)} --> {__format_timestamp(word.end)}\n"
                    )
                    f.write(f"{word.word.strip()}\n\n")

        else:
            for i, segment in track(
                enumerate(segments), description="Transcribing audio..."
            ):
                f.write(f"{i}\n")
                f.write(
                    f"{__format_timestamp(segment.start)} --> {__format_timestamp(segment.end)}\n"
                )
                f.write(f"{segment.text.strip()}\n\n")


class Subtitle(BaseModel):
    index: str = Field(..., description="Subtitle index")
    start: timedelta = Field(..., description="Start time in seconds")
    end: timedelta = Field(..., description="End time in seconds")
    content: str = Field(..., description="Subtitle text")


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
