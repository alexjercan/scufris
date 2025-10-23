import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

import ffmpeg
import ollama
from pydantic import BaseModel, Field
from rich.progress import track

from scufris.common import (
    Config,
    OllamaConfig,
    create_transcript,
    extract_audio,
    get_logger,
    setup_logger,
    with_stats,
)

PROMPT_TEMPLATE = """You are an expert summariser and video‐analysis assistant.

I will give you the full transcript of a video, in SRT format (timestamps + text). Your goals are:
1. Produce a **concise summary** of the video: highlight the key actions, decisions, demonstrations, and outcomes. Keep it to about ~250 words maximum.
2. Identify **key highlight segments** - select approximately **5 to 8** of the most important moments in the video (not every minute). For each highlight, provide the start-timestamp, end-timestamp, and a short description of what happens.
3. Provide a list of **3-5 key take-aways**: what the viewer should remember after watching the video.
4. Use clear, readable language: full sentences in the summary; short descriptive phrases for the highlights.

**Output Format:**
```json
{{
    "summary": "Concise summary of the video here...",
    "highlights": [
        {{
            "start": "HH:MM:SS,mmm",
            "end": "HH:MM:SS,mmm",
            "label": "Short description of the highlight."
        }}
    ],
    "key_takeaways": [
        "First key takeaway.",
        "Second key takeaway."
    ]
}}
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
```json
{{
    "summary": "In this demo, the presenter introduces a project with a React frontend and a Python Flask backend, supported by a PostgreSQL database. The folder structure is explored in VS Code, highlighting key components such as `App.js`, `server.py`, and `db.py`.",
    "highlights": [
        {{
            "start": "00:00:00,000",
            "end": "00:00:05,000",
            "label": "Introduction to the demo."
        }},
        {{
            "start": "00:00:15,000",
            "end": "00:00:30,000",
            "label": "Description of frontend and backend technologies."
        }},
        {{
            "start": "00:00:45,000",
            "end": "00:01:10,000",
            "label": "Highlighting main components in the project."
        }}
    ],
    "key_takeaways": [
        "The project uses a React frontend and Python Flask backend.",
        "The database is managed with PostgreSQL.",
        "Key files include `App.js`, `server.py`, and `db.py`."
    ]
}}
```

*Example 2:*
```srt
1
00:00:00,000 --> 00:00:08,000
Hi everyone — today I’ll walk you through our new machine-learning pipeline for image classification.

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
```json
{{
    "summary": "In this walkthrough, the presenter details a machine-learning pipeline for image classification. The process begins with loading and preprocessing a dataset of 10,000 labelled images, followed by constructing a convolutional neural network using PyTorch. The model is trained for 20 epochs, achieving a validation accuracy of approximately 92%.",
    "highlights": [
        {{
            "start": "00:00:00,000",
            "end": "00:00:08,000",
            "label": "Introduction to the machine-learning pipeline."
        }},
        {{
            "start": "00:00:20,000",
            "end": "00:00:40,000",
            "label": "Building the convolutional neural network in PyTorch."
        }},
        {{
            "start": "00:00:40,000",
            "end": "00:01:00,000",
            "label": "Training the model and achieving validation accuracy."
        }}
    ],
    "key_takeaways": [
        "The pipeline uses a dataset of 10,000 labelled images.",
        "A convolutional neural network is built using PyTorch.",
        "The model achieves approximately 92% validation accuracy after training."
    ]
}}
```

*Now*, here is the transcript you need to summarise:
```srt
{transcript}
```

Please produce the summary, sections, and key takeaways as per above.
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "highlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "format": "HH:MM:SS,mmm"},
                    "end": {"type": "string", "format": "HH:MM:SS,mmm"},
                    "label": {"type": "string"},
                },
                "required": ["start", "end", "label"],
            },
        },
        "key_takeaways": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "highlights", "key_takeaways"],
}


class Highlight(BaseModel):
    start: str = Field(alias="start", description="Start timestamp of the highlight.")
    end: str = Field(alias="end", description="End timestamp of the highlight.")
    label: str = Field(alias="label", description="Label of the highlight.")


class Summary(BaseModel):
    summary: str = Field(alias="summary", description="The main summary text.")
    highlights: List[Highlight] = Field(
        alias="highlights", description="List of highlights in the video."
    )
    key_takeaways: List[str] = Field(
        alias="key_takeaways", description="List of key takeaways from the video."
    )


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
        format=JSON_SCHEMA,
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(response.response)


def __clip_filename(section: Highlight, index: int) -> str:
    """Generate a filename for a clip based on the section and index."""

    start = section.start.replace(":", "-")
    end = section.end.replace(":", "-")
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
        summary_text = json.load(f)

    summary = Summary.model_validate(summary_text, by_alias=True)
    logger.debug(f"Parsed summary: {summary}")

    with open(run_path / "chapters.txt", "w", encoding="utf-8") as f:
        f.write("Chapters:\n")
        for section in summary.highlights:
            f.write(f"{section.start} - {section.label}\n")

    logger.info(f"Chapters saved as {run_path / 'chapters.txt'}")

    clip_files = []
    for i, section in track(
        enumerate(summary.highlights),
        description="Creating highlight clips...",
        total=len(summary.highlights),
    ):
        start = section.start
        end = section.end

        output_file = run_path / __clip_filename(section, i)
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


class Arguments(BaseModel):
    video_file: Path = Field(
        alias="video_file", description="Path to the input video file."
    )
    output: Path = Field(
        alias="output",
        default=Path("output.mp4"),
        description="Path to the output merged video file.",
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

    args = parser.parse_args()

    return Arguments(video_file=args.video_file, output=args.output)


def highlights() -> None:
    config = Config()
    args = parse_arguments()

    run_id = str(args.video_file.stem)
    logger = setup_logger(run_id)

    logger.info("Starting highlights process with run ID: %s", run_id)
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

    summary_path = Path(run_path, "summary.json")
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
