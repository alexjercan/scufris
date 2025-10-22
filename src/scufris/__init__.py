import argparse
import functools
import logging
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional

import ffmpeg
import ollama
import psutil
from faster_whisper import BatchedInferencePipeline, WhisperModel
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.logging import RichHandler
from rich.progress import track

CURRENT_DIR = Path.cwd()

PROMPT_TEMPLATE = """You are an expert summariser and video‐analysis assistant.

I will give you the full transcript of a video, in SRT format (timestamps + text). Your goals are:
1. Produce a **concise summary** of the video: highlight the key actions, decisions, demonstrations, and outcomes. Keep it to about ~250 words maximum.
2. Identify **key highlight segments** - select approximately **5 to 8** of the most important moments in the video (not every minute). For each highlight, provide the start-timestamp, end-timestamp, and a short description of what happens.
3. Provide a list of **3-5 key take-aways**: what the viewer should remember after watching the video.
4. Use clear, readable language: full sentences in the summary; short descriptive phrases for the highlights.

**Output Format:**
```
SUMMARY:
<One paragraph summary>

HIGHLIGHTS:
- <start> to <end>: <short description of this important moment>
- ... (5-8 items total)

KEY TAKEAWAYS:
- <Takeaway 1>
- <Takeaway 2>
- <Takeaway 3>
- ... (up to 5)
```

**Important instructions:**
- Only include highlight segments where something meaningful occurs (topic shift, demo step, decision, conclusion).
- Omit trivial or repetitive segments (e.g., long pauses, filler comments) unless they are indeed important.
- Do **not** list every section or minute of the video - this is about the *most significant moments*.
- Do **not** hallucinate or invent content: base everything strictly on the transcript.
- If section boundaries are unclear, estimate them reasonably.

**EXAMPLES:**

*Example 1:*
```srt
1
00:00:00,000 --> 00:00:05,000
Hello everybody, and welcome to today's demo.

2
00:00:05,000 --> 00:00:15,000
I'll begin by showing you the project architecture.

3
00:00:15,000 --> 00:00:30,000
We have a frontend in React and a backend in Python Flask, with a PostgreSQL database.

4
00:00:30,000 --> 00:00:45,000
Let's open VS Code and take a look at the folder structure...

5
00:00:45,000 --> 00:01:10,000
Here you can see the main components: `App.js`, `server.py`, and `db.py`.
```

*Output for Example 1:*
```
SUMMARY:
In this brief demo, the presenter introduces a web-application project built with a React frontend, Python Flask backend and a PostgreSQL database. After explaining the system's architecture, they open the code in VS Code, walk through the folder structure and key files (App.js, server.py, db.py), and set the stage for the upcoming live demo and deployment discussion.

HIGHLIGHTS:
- 00:00:00 to 00:00:05: Introduction & welcome
- 00:00:05 to 00:00:15: Overview of the architecture (React + Flask + PostgreSQL)
- 00:00:15 to 00:00:30: Explanation of frontend/back-end roles
- 00:00:30 to 00:00:45: Opening VS Code & navigating folder structure
- 00:00:45 to 00:01:10: Key files overview: App.js, server.py, db.py

KEY TAKEAWAYS:
1. The project uses React for the frontend, Flask for the backend, and PostgreSQL for data storage.
2. Understanding the folder structure and main code files is crucial before diving into the demo.
3. The stage is set for the live demo of how the app works and will soon be deployed.
```

*Example 2:*
```srt
1
00:00:00,000 --> 00:00:08,000
Hi everyone - today I’ll walk you through our new machine-learning pipeline for image classification.

2
00:00:08,000 --> 00:00:20,000
First we load the dataset of 10,000 labelled images in CSV format, preprocess them with normalization and augmentation…

3
00:00:20,000 --> 00:00:40,000
Then we build a convolutional neural network in PyTorch: two convolution layers, a max-pooling layer, and a softmax output for 5 classes.

4
00:00:40,000 --> 00:01:00,000
We train for 20 epochs, monitor validation accuracy, and achieve ~92% accuracy.
```

*Output for Example 2:*
```
SUMMARY:
The presenter guides the viewer through a machine-learning image classification project: starting from loading and preprocessing a dataset of 10,000 labelled images (including normalization and augmentation), they build a simple CNN in PyTorch (two conv layers, max-pooling, softmax for 5 classes), train it for 20 epochs, and achieve about 92% accuracy, followed by a discussion of next steps and deployment considerations.

HIGHLIGHTS:
- 00:00:00 to 00:00:08: Introduction to the ML pipeline
- 00:00:08 to 00:00:20: Dataset loading & preprocessing (10,000 images, augmentation)
- 00:00:20 to 00:00:40: Building the CNN model (architecture details)
- 00:00:40 to 00:01:00: Training results & accuracy (~92%)

KEY TAKEAWAYS:
1. The dataset contains 10,000 labelled images for a 5-class classification task.
2. Preprocessing (normalization + augmentation) is used to improve generalisation.
3. The CNN architecture is simple yet effective, achieving ~92% accuracy; the next step is deployment.
```

*Now*, here is the transcript you need to summarise:
```srt
{transcript}
```

Please produce the summary, sections, and key takeaways as per above.
"""


class Config(BaseSettings):
    """Main configuration for the application."""

    model_config = SettingsConfigDict(env_prefix="HIGHLIGHTS_")

    projects_dir: Path = Field(
        default=Path(CURRENT_DIR, "projects"),
        description="Directory to store projects.",
    )


class WhisperConfig(BaseModel):
    """Configuration for the Whisper model."""

    model_config = SettingsConfigDict(env_prefix="HIGHLIGHTS_WHISPER_")

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

    model_config = SettingsConfigDict(env_prefix="HIGHLIGHTS_OLLAMA_")

    model: str = Field(
        default="qwen3",
        description="Ollama model to use for summarization.",
    )
    think: bool = Field(
        default=True,
        description="Whether to enable 'think' mode in Ollama.",
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


def setup_logger(run_id: str) -> logging.Logger:
    """Setup a logger with a unique run ID."""

    class RunIDFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.run_id = run_id
            return True

    FORMAT = "[RUN %(run_id)s] %(message)s"

    logger = get_logger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(FORMAT, datefmt="[%X]")
    console_handler = RichHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addFilter(RunIDFilter())

    return logger


def get_logger() -> logging.Logger:
    """Get the logger instance."""
    return logging.getLogger("highlights")


def with_stats(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        process = psutil.Process(os.getpid())

        start_memory = process.memory_info().rss
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.debug(
            f"Function '{func.__name__}' executed in {elapsed_time:.2f} seconds."
        )

        end_memory = process.memory_info().rss
        memory_used = end_memory - start_memory
        memory_used = memory_used / (1024 * 1024)  # Convert to MB
        logger.debug(f"Function '{func.__name__}' used {memory_used:.2f} MB of memory.")

        return result

    return wrapper


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

    args = parser.parse_args()

    return Arguments(video_file=args.video_file, output=args.output)


@with_stats
def extract_audio(
    audio_path: Path, video_path: Path, logger: Optional[logging.Logger] = None
):
    """Extract audio from the given video file and save it as a WAV file."""

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
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=2000),
        beam_size=config.beam_size,
    )
    logger.debug(f"Whisper transcription arguments: {args}")

    if config.batch_size == 1:
        logger.debug("Using single inference pipeline for transcription.")
        segments, info = model.transcribe(str(audio_path), **args)
    else:
        logger.debug(
            f"Using batched inference pipeline for transcription with batch size {config.batch_size}."
        )
        batched_model = BatchedInferencePipeline(model)
        segments, info = batched_model.transcribe(
            str(audio_path),
            batch_size=config.batch_size,
            **args,
        )

    logger.debug(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )

    with open(transcript_path, "w", encoding="utf-8") as f:
        for i, segment in track(
            enumerate(segments), description="Transcribing audio..."
        ):
            f.write(f"{i}\n")
            f.write(
                f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
            )
            f.write(f"{segment.text.strip()}\n\n")


@with_stats
def create_summary(
    summary_path: Path,
    transcript_path: Path,
    logger: Optional[logging.Logger] = None,
    config: Optional[OllamaConfig] = None,
):
    """Create a summary of the transcript using Ollama."""

    logger = logger or get_logger()
    config = config or OllamaConfig()
    logger.debug(
        f"Creating summary for transcript: {transcript_path} to {summary_path} using config: {config}"
    )

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    prompt = PROMPT_TEMPLATE.format(transcript=transcript_text)
    prompt_path = summary_path.with_suffix(".prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    response = ollama.generate(
        model=config.model,
        prompt=prompt,
        think=config.think,
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(response.response)


class Section(BaseModel):
    start_timestamp: str = Field(description="Start timestamp of the section.")
    end_timestamp: str = Field(description="End timestamp of the section.")
    label: str = Field(description="Label of the section.")


class Summary(BaseModel):
    summary_text: str = Field(description="The main summary text.")
    sections: List[Section] = Field(description="List of sections in the video.")
    key_takeaways: List[str] = Field(
        description="List of key takeaways from the video."
    )


def parse_section_line(line: str) -> Optional[Section]:
    """Parse a section line into a Section object."""

    if not line.startswith("-"):
        return None

    try:
        line = line.lstrip("-").strip()
        timestamp_part, label = line.split(" - ", 1)
        start_timestamp, end_timestamp = timestamp_part.split(" to ", 1)

        return Section(
            start_timestamp=start_timestamp.strip(),
            end_timestamp=end_timestamp.strip(),
            label=label.strip(),
        )
    except ValueError:
        return None


def parse_summary(summary_text: str) -> Summary:
    """Parse the summary text into a Summary object."""

    lines = summary_text.splitlines()

    current_section = None
    summary_text_lines = []
    section_lines = []
    key_takeaways = []

    for line in lines:
        if line.strip().startswith("SUMMARY"):
            current_section = "summary"
            continue
        if line.strip().startswith("SECTIONS"):
            current_section = "sections"
            continue
        if line.strip().startswith("KEY TAKEAWAYS"):
            current_section = "key_takeaways"
            continue

        if current_section == "summary":
            summary_text_lines.append(line.strip())
        elif current_section == "sections":
            line = line.strip()
            if line:
                section_lines.append(line)
        elif current_section == "key_takeaways":
            key_takeaways.append(line.strip().lstrip("0123456789. ").strip())
        else:
            continue

    summary_text_combined = " ".join(summary_text_lines).strip()
    sections = [parse_section_line(line) for line in section_lines]

    return Summary(
        summary_text=summary_text_combined,
        sections=sections,
        key_takeaways=key_takeaways,
    )


def clip_filename(section: Section, index: int) -> str:
    """Generate a filename for a clip based on the section and index."""

    start = section.start_timestamp.replace(":", "-")
    end = section.end_timestamp.replace(":", "-")
    return f"clip_{index + 1:04d}_{start}_to_{end}.mp4"


@with_stats
def create_highlights(
    run_path: Path,
    video_path: Path,
    summary_path: Path,
    output_path: Path,
    logger: Optional[logging.Logger] = None,
):
    """Create highlights from the summary."""

    logger = logger or get_logger()
    logger.debug(
        f"Creating highlights for summary: {summary_path} in run path: {run_path}"
    )

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_text = f.read()

    summary = parse_summary(summary_text)
    logger.debug(f"Parsed summary: {summary}")

    with open(run_path / "chapters.txt", "w", encoding="utf-8") as f:
        f.write("Chapters:\n")
        for section in summary.sections:
            f.write(f"{section.start_timestamp} - {section.label}\n")

    logger.info(f"Chapters saved as {run_path / 'chapters.txt'}")

    clip_files = []
    for i, section in track(
        enumerate(summary.sections),
        description="Creating highlight clips...",
        total=len(summary.sections),
    ):
        start = section.start_timestamp
        end = section.end_timestamp

        output_file = run_path / clip_filename(section, i)
        clip_files.append(output_file)

        if output_file.exists():
            logger.debug(f"Clip already exists, skipping: {output_file}")
            continue

        (
            ffmpeg.input(str(video_path), ss=start, to=end)
            .output(
                str(output_file),
                vcodec="libx264",
                video_bitrate="4000k",
                acodec="aac",
                audio_bitrate="300k",
                y=None,
            )
            .overwrite_output()
            .run(quiet=True)
        )
        logger.debug(f"Created clip: {output_file} from {start} to {end}")

    merge_list_path = run_path / "merge_list.txt"
    with open(merge_list_path, "w", encoding="utf-8") as f:
        for clip in clip_files:
            f.write(f"file '{clip.resolve()}'\n")

    output_merged_path = run_path / "output_merged.mp4"

    (
        ffmpeg.input(str(merge_list_path), f="concat", safe=0)
        .output(
            str(output_merged_path),
            c="copy",
            y=None,
        )
        .overwrite_output()
        .run(quiet=True)
    )

    shutil.copyfile(output_merged_path, output_path)

    logger.info(f"Done! Merged video saved as {output_merged_path}")


def highlights() -> None:
    config = Config()
    args = parse_arguments()

    run_id = str(args.video_file.stem)
    logger = setup_logger(run_id)

    logger.info("Starting transcription process with run ID: %s", run_id)
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
        create_transcript(transcript_path, audio_path)
        logger.info(f"Created transcript at: {transcript_path}")
    else:
        logger.info(f"Cached transcript found at: {transcript_path}")

    summary_path = Path(run_path, "summary.txt")
    if not summary_path.exists():
        create_summary(summary_path, transcript_path)
        logger.info(f"Created summary at: {summary_path}")
    else:
        logger.info(f"Cached summary found at: {summary_path}")

    create_highlights(run_path, args.video_file, summary_path, args.output)

    print(f"Run ID: {run_id} finished!")
    print(f"Merged video path: {args.output}")
    print()
    with open(run_path / "chapters.txt", "r", encoding="utf-8") as f:
        chapters_text = f.read()
    print(chapters_text)
