import logging
from pathlib import Path
from typing import Optional

import ffmpeg

from .logging import get_logger, with_stats


@with_stats
def extract_audio(
    audio_path: Path, video_path: Path, logger: Optional[logging.Logger] = None
):
    """Extract audio from the given video file and save it as an audio file.

    Args:
        audio_path (Path): Path to save the extracted audio file.
        video_path (Path): Path to the input video file.
        logger (Optional[logging.Logger]): Logger instance for logging. If None, a default logger is used.
    """

    logger = logger or get_logger()
    logger.debug(f"Extracting audio from video: {video_path} to {audio_path}")

    (
        ffmpeg.input(video_path)
        .output(str(audio_path), ac=1, ar="16k")
        .overwrite_output()
        .run(quiet=True)
    )
